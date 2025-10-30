"""
Main script to generate all synthetic data
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
from supabase import create_client, Client

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.hospitals import generate_hospitals
from generators.admissions import generate_admissions_data
from generators.inventory import generate_inventory_data
from generators.events import format_events_for_db
import config

def save_to_json(data, filename):
    """Save data to JSON file for inspection"""
    output_dir = "data/generated"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"✓ Saved {len(data)} records to {filepath}")

def upload_to_supabase(supabase: Client, data: list, table_name: str, batch_size: int = 1000):
    """Upload data to Supabase in batches"""
    
    total = len(data)
    print(f"Uploading {total} records to '{table_name}'...")
    
    for i in range(0, total, batch_size):
        batch = data[i:i+batch_size]
        try:
            supabase.table(table_name).insert(batch).execute()
            print(f"  Uploaded batch {i//batch_size + 1} ({min(i+batch_size, total)}/{total})")
        except Exception as e:
            print(f"  Error uploading batch: {e}")
            # Save failed batch for inspection
            save_to_json(batch, f"failed_batch_{table_name}_{i}.json")
    
    print(f"✓ Completed upload to {table_name}\n")

def main():
    print("=" * 60)
    print("MedFlow AI - Synthetic Data Generation")
    print("=" * 60)
    print()
    
    # Step 1: Generate Hospitals
    print("Step 1: Generating hospital network...")
    hospitals = generate_hospitals(
        num_hospitals=config.NUM_HOSPITALS,
        regions=config.REGIONS,
        size_distribution=config.HOSPITAL_SIZES,
        specializations=config.SPECIALIZATIONS
    )
    print(f"✓ Generated {len(hospitals)} hospitals across {len(config.REGIONS)} regions")
    save_to_json(hospitals, "hospitals.json")
    print()
    
    # Step 2: Generate Patient Admissions
    print("Step 2: Generating patient admission data...")
    print(f"  Time period: {config.START_DATE.date()} to {config.END_DATE.date()}")
    print(f"  Total days: {config.TOTAL_DAYS}")
    
    admissions_data = generate_admissions_data(
        hospitals=hospitals,
        start_date=config.START_DATE,
        total_days=config.TOTAL_DAYS,
        events=config.OUTBREAK_EVENTS,
        seasonal_factors=config.SEASONAL_FACTORS,
        weekly_pattern=config.WEEKLY_PATTERN
    )
    print(f"✓ Generated {len(admissions_data)} admission records")
    save_to_json(admissions_data[:1000], "admissions_sample.json")  # Save sample
    print()
    
    # Step 3: Generate Resource Inventory
    print("Step 3: Generating resource inventory history...")
    print(f"  Tracking {len(config.RESOURCES)} resource types")
    
    inventory_history, current_inventory = generate_inventory_data(
        hospitals=hospitals,
        admissions_data=admissions_data,
        resources=config.RESOURCES,
        start_date=config.START_DATE,
        total_days=config.TOTAL_DAYS,
        events=config.OUTBREAK_EVENTS,
        supply_disruptions=config.SUPPLY_DISRUPTIONS
    )
    print(f"✓ Generated {len(inventory_history)} inventory history records")
    print(f"✓ Generated {len(current_inventory)} current inventory records")
    save_to_json(inventory_history[:1000], "inventory_history_sample.json")
    save_to_json(current_inventory, "current_inventory.json")
    print()
    
    # Step 4: Format Events
    print("Step 4: Formatting events...")
    events_data = format_events_for_db(
        outbreak_events=config.OUTBREAK_EVENTS,
        supply_disruptions=config.SUPPLY_DISRUPTIONS
    )
    print(f"✓ Generated {len(events_data)} event records")
    save_to_json(events_data, "events.json")
    print()
    
    # Step 5: Generate Statistics
    print("Step 5: Generating data statistics...")
    stats = {
        "generation_date": datetime.now().isoformat(),
        "hospitals": len(hospitals),
        "regions": len(config.REGIONS),
        "time_period": {
            "start": config.START_DATE.date().isoformat(),
            "end": config.END_DATE.date().isoformat(),
            "days": config.TOTAL_DAYS
        },
        "admissions_records": len(admissions_data),
        "inventory_history_records": len(inventory_history),
        "current_inventory_records": len(current_inventory),
        "events": len(events_data),
        "resource_types": list(config.RESOURCES.keys()),
        "hospitals_by_region": {},
        "avg_daily_admissions": sum(a["total_admissions"] for a in admissions_data) / len(admissions_data)
    }
    
    # Count hospitals by region
    for hospital in hospitals:
        region = hospital["region"]
        stats["hospitals_by_region"][region] = stats["hospitals_by_region"].get(region, 0) + 1
    
    save_to_json(stats, "generation_stats.json")
    print("✓ Statistics saved")
    print()
    
    # Step 6: Upload to Supabase (optional)
    upload = input("Upload data to Supabase? (yes/no): ").strip().lower()
    
    if upload == "yes":
        print("\nConnecting to Supabase...")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            print("✗ Supabase credentials not found in .env file")
            return
        
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✓ Connected to Supabase\n")
        
        # Fetch resource type IDs for mapping
        try:
            rt_resp = supabase.table("resource_types").select("id,name").execute()
            name_to_id = {row["name"]: row["id"] for row in rt_resp.data}
        except Exception as e:
            print(f"✗ Failed to fetch resource_types: {e}")
            return

        # Transform hospitals: drop non-schema fields
        transformed_hospitals = []
        for h in hospitals:
            new_h = {k: v for k, v in h.items() if k != "size"}
            transformed_hospitals.append(new_h)

        # Transform admissions: convert dates to ISO strings
        transformed_admissions = []
        for rec in admissions_data:
            new_rec = dict(rec)
            if isinstance(new_rec.get("admission_date"), (datetime,)):
                new_rec["admission_date"] = new_rec["admission_date"].date().isoformat()
            else:
                # date object -> isoformat
                try:
                    new_rec["admission_date"] = new_rec["admission_date"].isoformat()
                except Exception:
                    pass
            transformed_admissions.append(new_rec)

        # Transform inventory datasets to use resource_type_id
        transformed_current_inventory = []
        for rec in current_inventory:
            rt_name = rec.get("resource_type")
            if rt_name not in name_to_id:
                print(f"✗ Unknown resource_type '{rt_name}' in current_inventory; aborting upload")
                return
            new_rec = {
                "hospital_id": rec["hospital_id"],
                "resource_type_id": name_to_id[rt_name],
                "quantity": rec["quantity"],
                "reserved_quantity": rec.get("reserved_quantity", 0)
            }
            transformed_current_inventory.append(new_rec)

        transformed_inventory_history = []
        for rec in inventory_history:
            rt_name = rec.get("resource_type")
            if rt_name not in name_to_id:
                print(f"✗ Unknown resource_type '{rt_name}' in inventory_history; aborting upload")
                return
            new_rec = {
                "hospital_id": rec["hospital_id"],
                "resource_type_id": name_to_id[rt_name],
                "date": rec["date"].isoformat() if hasattr(rec.get("date"), "isoformat") else rec.get("date"),
                "quantity": rec["quantity"],
                "consumption": rec.get("consumption", 0),
                "resupply": rec.get("resupply", 0)
            }
            transformed_inventory_history.append(new_rec)

        # Transform events: ensure dates are ISO strings
        transformed_events = []
        for ev in events_data:
            new_ev = dict(ev)
            sd = new_ev.get("start_date")
            ed = new_ev.get("end_date")
            if hasattr(sd, "isoformat"):
                new_ev["start_date"] = sd.isoformat()
            if hasattr(ed, "isoformat"):
                new_ev["end_date"] = ed.isoformat()
            transformed_events.append(new_ev)

        # Upload in sequence
        upload_to_supabase(supabase, transformed_hospitals, "hospitals")
        upload_to_supabase(supabase, transformed_admissions, "patient_admissions")
        upload_to_supabase(supabase, transformed_current_inventory, "resource_inventory")
        upload_to_supabase(supabase, transformed_inventory_history, "inventory_history")
        upload_to_supabase(supabase, transformed_events, "events")
        
        print("=" * 60)
        print("✓ All data uploaded successfully!")
        print("=" * 60)
    else:
        print("\nData saved locally. To upload later, run this script again.")
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)
    print(f"\nGenerated:")
    print(f"  • {len(hospitals)} hospitals")
    print(f"  • {len(admissions_data):,} admission records")
    print(f"  • {len(inventory_history):,} inventory history records")
    print(f"  • {len(current_inventory)} current inventory snapshots")
    print(f"  • {len(events_data)} events")
    print(f"\nFiles saved to: data/generated/")

if __name__ == "__main__":
    main()