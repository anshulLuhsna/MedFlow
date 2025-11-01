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
    """Calculate daily resource consumption

    IMPORTANT: For ML training, consumption must have:
    1. Strong correlation with admissions (predictable signal)
    2. Sufficient variation (not mostly zeros)
    3. Realistic noise level (10-20%)
    """

    base_consumption = 0

    if resource_type == "ventilators":
        # FIXED v2: Ventilators track cumulative usage (ventilator-days)
        # For ML training, we need to track DAILY USAGE not just new allocations
        #
        # Model: Ventilators are occupied continuously for multiple days
        # At any time, there are ongoing patients + new patients
        # This creates a more stable, predictable signal

        if icu_admissions > 0:
            # Rate: 60% of ICU patients need mechanical ventilation (realistic for severe cases)
            # Duration: Patients stay on ventilators for 4-7 days average
            ventilator_rate = 0.60
            avg_duration_days = 5.0

            # NEW APPROACH: Track cumulative daily usage
            # Daily usage = new patients + existing patients still on ventilators
            # Simplified: ICU_admissions * rate * duration approximates daily census
            new_patients_needing_vent = icu_admissions * ventilator_rate

            # Expected daily ventilator usage (cumulative)
            # This is higher and more stable than just new allocations
            expected = new_patients_needing_vent * (avg_duration_days / 2)  # Average ongoing usage

            # Add variation
            base_consumption = max(0, int(expected + random.gauss(0, max(1.0, expected * 0.15))))

            # Ensure minimum usage when ICU exists
            if icu_admissions >= 2 and base_consumption == 0:
                base_consumption = int(icu_admissions * 0.3)  # At least 30% of ICU patients
            elif icu_admissions >= 1 and base_consumption == 0:
                base_consumption = 1 if random.random() < 0.6 else 0

            # Add small baseline usage even with low ICU (ongoing patients from previous days)
            if base_consumption < 2 and random.random() < 0.3:
                base_consumption += 1
        else:
            # No ICU patients = minimal ventilator usage (could have ongoing from previous days)
            base_consumption = 1 if random.random() < 0.1 else 0
    
    elif resource_type == "o2_cylinders":
        # FIXED: O2 cylinders are consumed by many patients
        # Both ICU and regular respiratory patients
        # Strong correlation with admissions (good for ML)

        # ICU patients: 2-3 cylinders per patient per day
        # Regular respiratory patients: ~20% of admissions, 0.5-1 cylinder per day
        icu_consumption = icu_admissions * random.uniform(2.0, 3.0)
        regular_consumption = admissions * 0.20 * random.uniform(0.5, 1.0)

        base_consumption = int(icu_consumption + regular_consumption)

        # Ensure minimum consumption if patients exist (realistic floor)
        if admissions > 0 and base_consumption == 0:
            base_consumption = max(1, int(admissions * 0.1))

    elif resource_type == "beds":
        # FIXED: Beds track occupancy (census), not consumption
        # This is predictable from admissions with some persistence
        # Average length of stay: 3-5 days
        # Daily "consumption" = new occupied bed-days
        avg_los = 3.5  # Average length of stay
        base_consumption = int(admissions * resource_config["consumption_rate"] * (avg_los / 7))

        # Ensure realistic floor
        if admissions > 0 and base_consumption == 0:
            base_consumption = int(admissions * 0.5)  # At least half of admissions

    elif resource_type == "medications":
        # FIXED: Medications have strong signal from admissions
        # All patients consume medications, with variation by severity
        # Rate: 3-7 doses per patient per day (average 5 from config)
        variation = random.uniform(0.8, 1.2)
        base_consumption = int(admissions * resource_config["consumption_rate"] * variation)

        # Ensure realistic minimum
        if admissions > 0 and base_consumption == 0:
            base_consumption = max(1, admissions)

    elif resource_type == "ppe":
        # FIXED: PPE consumption from staff (scales with patient load)
        # Staff-to-patient ratio ~1:4, each staff uses 2-3 sets per shift
        # This creates predictable signal from admissions
        staff_ratio = 0.25  # 1 staff per 4 patients
        ppe_per_staff = 2.5  # Sets per shift

        variation = random.uniform(0.9, 1.1)
        base_consumption = int(admissions * staff_ratio * ppe_per_staff * variation)

        # Ensure realistic minimum
        if admissions > 0 and base_consumption == 0:
            base_consumption = max(1, int(admissions * 0.3))
    
    # Check for events that affect this resource
    for event in events:
        if is_in_event(date, event):
            if region in event.get("affected_regions", []):
                multiplier = event.get("resource_multiplier", {}).get(resource_type, 1.0)
                base_consumption = int(base_consumption * multiplier)

    # Add noise for realism (10% noise level - good for ML training)
    # Too little noise = model overfits
    # Too much noise = model can't learn signal
    # 10% is the sweet spot for medical data
    consumption_with_noise = add_noise(base_consumption, noise_level=0.10)

    # Ensure non-negative (consumption can't be negative)
    final_consumption = max(0, int(consumption_with_noise))

    return final_consumption

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
    
    # For resources with low consumption (like ventilators), ensure periodic resupply
    # to maintain inventory variation
    if resource_type == "ventilators":
        # More frequent but smaller resupplies for ventilators
        if random.random() < 0.4:  # 40% chance of routine resupply
            # Small resupply to replace any consumption or maintain levels
            return max(0, int(consumption * random.uniform(0.8, 1.5)) + (1 if random.random() < 0.3 else 0))
    
    # Random small resupply even when not critically low (routine procurement)
    # Increased frequency for better variation
    if random.random() < 0.3:  # 30% chance (increased from 20%)
        return int(consumption * random.uniform(0.5, 1.2))
    
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
                
                # Ensure minimum variation for learning (especially for low-consumption resources)
                # Add small random adjustments to prevent completely static inventory
                if resource_type in ["ventilators", "o2_cylinders"]:
                    # Once every ~10 days, add small random adjustment (±1 unit)
                    if day_idx % 10 == 0 and random.random() < 0.5:
                        adjustment = random.choice([-1, 0, 1])
                        current_stock = max(0, current_stock + adjustment)
                
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