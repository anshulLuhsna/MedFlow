"""
Generate resource inventory and consumption patterns
"""

from datetime import timedelta
import random
from .patterns import is_in_event, add_noise

def calculate_base_inventory(hospital_capacity: int, resource_config: dict) -> int:
    """Calculate baseline inventory for a resource type"""
    baseline = hospital_capacity * resource_config["baseline_per_bed"]
    # Add some random variation (some hospitals stock more)
    variation = random.uniform(0.8, 1.3)
    return int(baseline * variation)

def calculate_resource_consumption(
    admissions: int,
    icu_admissions: int,
    resource_type: str,
    resource_config: dict,
    date,
    region: str,
    events: list
) -> int:
    """Calculate daily resource consumption"""
    
    base_consumption = 0
    
    if resource_type == "ventilators":
        # Mainly ICU patients
        base_consumption = int(icu_admissions * resource_config["consumption_rate"])
    
    elif resource_type == "o2_cylinders":
        # Both ICU and regular patients
        base_consumption = int((icu_admissions * 2) + (admissions * 0.3))
    
    elif resource_type == "beds":
        # Beds are "consumed" as occupancy
        base_consumption = int(admissions * resource_config["consumption_rate"])
    
    elif resource_type == "medications":
        # All patients consume medications
        base_consumption = int(admissions * resource_config["consumption_rate"])
    
    elif resource_type == "ppe":
        # Staff use PPE, proportional to patient load
        base_consumption = int(admissions * resource_config["consumption_rate"])
    
    # Check for events that affect this resource
    for event in events:
        if is_in_event(date, event):
            if region in event.get("affected_regions", []):
                multiplier = event.get("resource_multiplier", {}).get(resource_type, 1.0)
                base_consumption = int(base_consumption * multiplier)
    
    # Add noise
    return add_noise(base_consumption, noise_level=0.10)

def calculate_resupply(
    current_stock: int,
    consumption: int,
    base_inventory: int,
    resource_type: str
) -> int:
    """Calculate resupply amount (simulating procurement)"""
    
    # Target: maintain around 70-90% of base inventory
    target_stock = int(base_inventory * random.uniform(0.7, 0.9))
    
    # After consumption, would we be below target?
    projected_stock = current_stock - consumption
    
    if projected_stock < target_stock:
        # Resupply to get close to target
        resupply = target_stock - projected_stock
        # Add some randomness (deliveries not always exact)
        resupply = int(resupply * random.uniform(0.8, 1.2))
        return max(0, resupply)
    
    # Random small resupply even when not critically low (routine procurement)
    if random.random() < 0.2:  # 20% chance
        return int(consumption * random.uniform(0.5, 1.0))
    
    return 0

def apply_supply_disruption(
    resupply: int,
    date,
    region: str,
    resource_type: str,
    disruptions: list
) -> int:
    """Apply supply disruption effects"""
    
    for disruption in disruptions:
        if is_in_event(date, disruption):
            if region in disruption.get("affected_regions", []):
                if resource_type in disruption.get("affected_resources", []):
                    reduction = disruption.get("supply_reduction", 0.5)
                    resupply = int(resupply * (1 - reduction))
    
    return resupply

def generate_inventory_data(
    hospitals: list,
    admissions_data: list,
    resources: dict,
    start_date,
    total_days: int,
    events: list,
    supply_disruptions: list
) -> tuple:
    """
    Generate inventory history for all hospitals and resources
    
    Returns:
        (inventory_history, current_inventory)
    """
    
    inventory_history = []
    current_inventory = []
    
    # Group admissions by hospital and date for easy lookup
    admissions_lookup = {}
    for admission in admissions_data:
        key = (admission["hospital_id"], admission["admission_date"])
        admissions_lookup[key] = admission
    
    for hospital in hospitals:
        hospital_id = hospital["id"]
        region = hospital["region"]
        capacity = hospital["capacity_beds"]
        
        # Initialize inventory for each resource type
        resource_stocks = {}
        
        for resource_type, resource_config in resources.items():
            # Starting inventory
            base_inventory = calculate_base_inventory(capacity, resource_config)
            current_stock = base_inventory
            resource_stocks[resource_type] = {
                "base": base_inventory,
                "current": current_stock
            }
            
            # Generate daily history
            for day_idx in range(total_days):
                current_date = start_date + timedelta(days=day_idx)
                
                # Get admission data for this day
                admission_key = (hospital_id, current_date.date())
                admission = admissions_lookup.get(admission_key, {
                    "total_admissions": 0,
                    "icu_admissions": 0
                })
                
                # Calculate consumption
                consumption = calculate_resource_consumption(
                    admissions=admission.get("total_admissions", 0),
                    icu_admissions=admission.get("icu_admissions", 0),
                    resource_type=resource_type,
                    resource_config=resource_config,
                    date=current_date,
                    region=region,
                    events=events
                )
                
                # Calculate resupply
                resupply = calculate_resupply(
                    current_stock=current_stock,
                    consumption=consumption,
                    base_inventory=base_inventory,
                    resource_type=resource_type
                )
                
                # Apply supply disruptions
                resupply = apply_supply_disruption(
                    resupply=resupply,
                    date=current_date,
                    region=region,
                    resource_type=resource_type,
                    disruptions=supply_disruptions
                )
                
                # Update stock
                current_stock = current_stock - consumption + resupply
                current_stock = max(0, current_stock)  # Can't go negative
                
                # Record history
                inventory_history.append({
                    "hospital_id": hospital_id,
                    "resource_type": resource_type,
                    "date": current_date.date(),
                    "quantity": current_stock,
                    "consumption": consumption,
                    "resupply": resupply
                })
                
                # Update current stock for next day
                resource_stocks[resource_type]["current"] = current_stock
        
        # Record final inventory state
        for resource_type, stock_info in resource_stocks.items():
            current_inventory.append({
                "hospital_id": hospital_id,
                "resource_type": resource_type,
                "quantity": stock_info["current"],
                "reserved_quantity": 0
            })
    
    return inventory_history, current_inventory