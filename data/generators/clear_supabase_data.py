"""
Script to clear synthetic data from Supabase tables before regenerating.

This clears data in the correct order to respect foreign key constraints.
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def clear_table_data(supabase: Client, table_name: str, description: str = None) -> int:
    """
    Delete all rows from a table.
    
    Returns:
        Number of rows deleted
    """
    if description:
        print(f"  Clearing {description}...", end=" ")
    else:
        print(f"  Clearing {table_name}...", end=" ")
    
    try:
        # Delete all rows (Supabase will respect FK constraints)
        response = supabase.table(table_name).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        
        # Count is not directly available, so we'll just report success
        print("✓")
        return 1  # Success indicator
    except Exception as e:
        # If error is about empty table, that's fine
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            print("✓ (empty)")
            return 0
        else:
            print(f"✗ Error: {e}")
            return 0


def main():
    print("=" * 60)
    print("MedFlow AI - Clear Supabase Data")
    print("=" * 60)
    print()
    print("⚠ WARNING: This will delete all synthetic data from:")
    print("  • inventory_history")
    print("  • patient_admissions")
    print("  • resource_inventory")
    print("  • events")
    print("  • hospitals (optional)")
    print()
    print("This will NOT delete:")
    print("  • resource_types (reference table)")
    print("  • resource_requests, allocations (operational data)")
    print("  • user_interactions, preference_profiles (user data)")
    print()
    
    confirm = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return
    
    print("\nConnecting to Supabase...")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        print("✗ Supabase credentials not found in .env file")
        return
    
    supabase: Client = create_client(supabase_url, supabase_key)
    print("✓ Connected to Supabase\n")
    
    # Clear tables in order (child tables first due to FK constraints)
    print("Clearing data tables...")
    
    # 1. Clear inventory_history (references hospitals)
    clear_table_data(supabase, "inventory_history", "inventory history")
    
    # 2. Clear patient_admissions (references hospitals)
    clear_table_data(supabase, "patient_admissions", "patient admissions")
    
    # 3. Clear resource_inventory (references hospitals)
    clear_table_data(supabase, "resource_inventory", "current inventory")
    
    # 4. Clear events (no FK, but synthetic data)
    clear_table_data(supabase, "events", "events")
    
    # 5. Optional: Clear hospitals (will cascade delete related data)
    print()
    clear_hospitals = input("Delete hospitals table? (yes/no): ").strip().lower()
    if clear_hospitals == "yes":
        clear_table_data(supabase, "hospitals", "hospitals")
        print("  ⚠ Hospitals deleted - new hospitals will be created with different IDs")
    else:
        print("  ✓ Keeping existing hospitals")
    
    # Optional: Clear shortage_predictions if they exist
    try:
        clear_table_data(supabase, "shortage_predictions", "shortage predictions (ML outputs)")
    except:
        pass  # Table might not exist
    
    print()
    print("=" * 60)
    print("✓ Data clearing complete!")
    print("=" * 60)
    print()
    print("You can now regenerate synthetic data by running:")
    print("  python data/generators/generate_synthetic_data.py")


if __name__ == "__main__":
    main()

