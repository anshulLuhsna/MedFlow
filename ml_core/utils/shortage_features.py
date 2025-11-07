"""
Shortage Feature Engineering
Engineers 20 features from raw data for shortage risk detection
"""

import pandas as pd
import numpy as np
from typing import Optional
from typing import Dict, List
from datetime import datetime, timedelta


def engineer_shortage_features(
    current_inventory: pd.DataFrame,
    demand_predictions: pd.DataFrame,
    admissions_history: pd.DataFrame,
    hospital_info: pd.DataFrame,
    regional_inventory: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Engineer features for shortage detection

    Args:
        current_inventory: Columns: hospital_id, resource_type, quantity,
                          last_resupply_date, capacity, region
        demand_predictions: Columns: hospital_id, resource_type, day,
                           predicted_consumption
        admissions_history: Columns: hospital_id, date, total_admissions,
                           icu_admissions, emergency_admissions
        hospital_info: Columns: hospital_id, region, capacity, specialization,
                       latitude, longitude

    Returns:
        DataFrame with hospital_id, resource_type, and 20 engineered features
    """
    # Use regional_inventory if provided, otherwise use current_inventory for regional calculations
    inventory_for_regional = regional_inventory if regional_inventory is not None else current_inventory
    
    features_list = []

    # For each hospital-resource pair
    for idx, row in current_inventory.iterrows():
        hospital_id = row['hospital_id']
        
        # Handle resource_type: check multiple possible column formats
        # Supabase might return nested dict or flattened column
        resource_type = None
        if 'resource_type' in row:
            resource_type = row['resource_type']
        elif 'resource_type_id' in row:
            # If we only have ID, we'll need to map it
            resource_type_id = row['resource_type_id']
            # Try to get from nested resource_types dict
            if isinstance(row.get('resource_types'), dict):
                resource_type = row['resource_types'].get('name')
            elif isinstance(row.get('resource_type'), dict):
                resource_type = row['resource_type'].get('name')
            # If still None, use the ID as fallback (will be handled later)
            if resource_type is None:
                resource_type = str(resource_type_id)
        elif isinstance(row.get('resource_types'), dict):
            resource_type = row['resource_types'].get('name')
        
        if resource_type is None:
            # Skip if we can't determine resource type
            continue

        # Get related data
        demand = demand_predictions[
            (demand_predictions['hospital_id'] == hospital_id) &
            (demand_predictions['resource_type'] == resource_type)
        ]
        
        # Check if demand DataFrame is empty or missing 'day' column
        if demand.empty:
            # Skip this hospital-resource pair if no demand predictions
            continue
        
        # Validate required columns
        if 'day' not in demand.columns:
            print(f"[Shortage Features] Warning: Missing 'day' column for hospital {hospital_id}, resource {resource_type}")
            print(f"[Shortage Features] Available columns: {demand.columns.tolist()}")
            continue
        
        # Sort by day
        demand = demand.sort_values('day')

        # Get admissions for this hospital (last 30 days)
        hospital_admissions = admissions_history[
            admissions_history['hospital_id'] == hospital_id
        ]
        
        # Check if 'date' or 'admission_date' column exists
        date_col = 'admission_date' if 'admission_date' in hospital_admissions.columns else 'date'
        admissions = hospital_admissions.sort_values(date_col).tail(30)  # Last 30 days

        hospital = hospital_info[
            hospital_info['hospital_id'] == hospital_id
        ]

        if hospital.empty:
            continue

        hospital = hospital.iloc[0]

        # Calculate predicted_demand_7d (sum of first 7 days)
        predicted_demand_7d = demand['predicted_consumption'].iloc[:7].sum() if len(demand) >= 7 else demand['predicted_consumption'].sum()
        
        # Extract features
        features = {
            'hospital_id': hospital_id,
            'resource_type': resource_type,
            
            # Required columns for shortage_detector.detect_shortages
            'stock_level': row['quantity'],  # Current stock level
            'predicted_demand_7d': predicted_demand_7d,  # 7-day predicted demand

            # Category 1: Stock-Demand Ratios (4 features)
            'stock_demand_ratio': calculate_stock_demand_ratio(row, demand, days=7),
            'stock_demand_ratio_14d': calculate_stock_demand_ratio(row, demand, days=14),
            'stock_capacity_ratio': row['quantity'] / row.get('capacity', row.get('max_capacity', 100)) if row.get('capacity', row.get('max_capacity', 100)) > 0 else 0,
            'demand_capacity_ratio': calculate_demand_capacity_ratio(row, demand),

            # Category 2: Time-Based Indicators (4 features)
            'days_of_supply': calculate_days_of_supply(row, demand),
            'days_since_resupply': calculate_days_since_resupply(row),
            'days_to_critical': calculate_days_to_critical(row, demand),
            'predicted_stockout_day': calculate_stockout_day(row, demand),

            # Category 3: Consumption Velocity (4 features)
            'consumption_trend_7d': calculate_consumption_trend(demand, days=7),
            'consumption_volatility': calculate_consumption_volatility(demand, days=7),
            'consumption_acceleration': calculate_acceleration(demand),
            'predicted_demand_change': calculate_demand_change(demand),

            # Category 4: Regional Context (3 features)
            # Use regional_inventory (unfiltered) for regional calculations
            'regional_avg_stock': calculate_regional_avg(inventory_for_regional, hospital.get('region', 'unknown'), resource_type),
            'regional_transfer_availability': calculate_transfer_availability(
                inventory_for_regional, hospital_info, hospital_id, hospital.get('region', 'unknown'), resource_type
            ),
            'isolation_score': calculate_isolation(hospital_info, hospital_id),

            # Category 5: Admission Patterns (3 features)
            'admission_trend_7d': calculate_trend(admissions['total_admissions'], days=7) if not admissions.empty else 0,
            'icu_admission_ratio': calculate_icu_ratio(admissions) if not admissions.empty else 0,
            'emergency_admission_spike': calculate_spike(admissions['emergency_admissions']) if not admissions.empty else 0,

            # Category 6: Resource-Specific (2 features)
            'resource_criticality': get_resource_criticality(resource_type),
            'consumption_per_admission': calculate_consumption_per_admission(demand, admissions)
        }

        features_list.append(features)

    return pd.DataFrame(features_list)


# Category 1: Stock-Demand Ratios
def calculate_stock_demand_ratio(inventory_row, demand_df, days=7):
    """Calculate stock to predicted demand ratio"""
    if demand_df.empty or len(demand_df) < days:
        return 999.0  # Very high ratio (no shortage)

    predicted_demand = demand_df['predicted_consumption'].iloc[:days].sum()
    if predicted_demand == 0:
        return 999.0
    return inventory_row['quantity'] / predicted_demand


def calculate_demand_capacity_ratio(inventory_row, demand_df):
    """Calculate 7-day demand relative to capacity"""
    if demand_df.empty or len(demand_df) < 7:
        return 0.0

    demand_7d = demand_df['predicted_consumption'].iloc[:7].sum()
    capacity = inventory_row.get('capacity', inventory_row.get('max_capacity', 100))
    if capacity == 0:
        return 999.0
    return demand_7d / capacity


# Category 2: Time-Based Indicators
def calculate_days_of_supply(inventory_row, demand_df):
    """Calculate days until stockout"""
    if demand_df.empty:
        return 999.0

    avg_daily_demand = demand_df['predicted_consumption'].iloc[:7].mean() if len(demand_df) >= 7 else demand_df['predicted_consumption'].mean()

    if avg_daily_demand == 0:
        return 999.0
    return inventory_row['quantity'] / avg_daily_demand


def calculate_days_since_resupply(inventory_row):
    """Calculate days since last resupply"""
    try:
        # Try multiple possible column names
        last_resupply_date = inventory_row.get('last_resupply_date') or inventory_row.get('last_updated') or inventory_row.get('last_resupply')
        if last_resupply_date is None:
            return 999  # Unknown
        
        last_resupply = pd.to_datetime(last_resupply_date)
        days = (pd.Timestamp.now() - last_resupply).days
        return max(0, days)
    except:
        return 999  # Unknown


def calculate_days_to_critical(inventory_row, demand_df, critical_threshold=0.2):
    """Calculate days until critical shortage"""
    capacity = inventory_row.get('capacity', inventory_row.get('max_capacity', 100))
    critical_stock = capacity * critical_threshold
    current_stock = inventory_row['quantity']

    if current_stock <= critical_stock:
        return 0

    if demand_df.empty:
        return 999

    # Cumulative consumption until critical
    cumulative = 0
    for day, consumption in enumerate(demand_df['predicted_consumption']):
        cumulative += consumption
        if current_stock - cumulative <= critical_stock:
            return day + 1

    return 14  # Beyond forecast horizon


def calculate_stockout_day(inventory_row, demand_df):
    """Calculate exact day of stockout"""
    current_stock = inventory_row['quantity']

    if demand_df.empty:
        return 999

    cumulative = 0
    for day, consumption in enumerate(demand_df['predicted_consumption']):
        cumulative += consumption
        if cumulative >= current_stock:
            return day + 1

    return 15  # Beyond forecast horizon


# Category 3: Consumption Velocity
def calculate_consumption_trend(demand_df, days=7):
    """Calculate consumption trend slope"""
    if demand_df.empty or len(demand_df) < 2:
        return 0

    recent = demand_df['predicted_consumption'].iloc[:min(days, len(demand_df))].values

    if len(recent) < 2:
        return 0

    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    return float(slope)


def calculate_consumption_volatility(demand_df, days=7):
    """Calculate consumption standard deviation"""
    if demand_df.empty:
        return 0

    recent = demand_df['predicted_consumption'].iloc[:min(days, len(demand_df))].values
    return float(np.std(recent))


def calculate_acceleration(demand_df):
    """Calculate 2nd derivative (acceleration)"""
    if demand_df.empty or len(demand_df) < 3:
        return 0

    recent = demand_df['predicted_consumption'].iloc[:7].values

    if len(recent) < 3:
        return 0

    # Calculate 2nd derivative
    first_deriv = np.gradient(recent)
    second_deriv = np.gradient(first_deriv)
    return float(second_deriv[-1])


def calculate_demand_change(demand_df):
    """Calculate % change from day 1 to day 7"""
    if demand_df.empty or len(demand_df) < 7:
        return 0

    day1 = demand_df['predicted_consumption'].iloc[0]
    day7 = demand_df['predicted_consumption'].iloc[6]

    if day1 == 0:
        return 0

    return (day7 - day1) / day1


# Category 4: Regional Context
def calculate_regional_avg(inventory_df, region, resource_type):
    """Calculate average stock in region for this resource"""
    regional = inventory_df[
        (inventory_df['region'] == region) &
        (inventory_df['resource_type'] == resource_type)
    ]

    if regional.empty:
        return 0

    return float(regional['quantity'].mean())


def calculate_transfer_availability(inventory_df, hospital_info_df, hospital_id, region, resource_type):
    """Calculate available surplus in region"""
    # Get other hospitals in region
    regional_hospitals = hospital_info_df[
        (hospital_info_df['region'] == region) &
        (hospital_info_df['hospital_id'] != hospital_id)
    ]['hospital_id'].values

    if len(regional_hospitals) == 0:
        return 0

    # Calculate surplus (quantity > 0.7 * capacity)
    surplus = 0
    for h_id in regional_hospitals:
        h_inventory = inventory_df[
            (inventory_df['hospital_id'] == h_id) &
            (inventory_df['resource_type'] == resource_type)
        ]

        for _, row in h_inventory.iterrows():
            capacity = row.get('capacity', row.get('max_capacity', 100))
            if capacity > 0 and row['quantity'] > 0.7 * capacity:
                surplus += (row['quantity'] - 0.7 * capacity)

    return float(surplus)


def calculate_isolation(hospital_info_df, hospital_id):
    """Calculate distance to nearest hospital"""
    hospital = hospital_info_df[hospital_info_df['hospital_id'] == hospital_id]

    if hospital.empty:
        return 0

    hospital = hospital.iloc[0]

    min_distance = 999.0
    for _, other in hospital_info_df.iterrows():
        if other['hospital_id'] == hospital_id:
            continue

        # Euclidean distance (simplified)
        distance = np.sqrt(
            (hospital['latitude'] - other['latitude'])**2 +
            (hospital['longitude'] - other['longitude'])**2
        )

        if distance < min_distance:
            min_distance = distance

    return float(min_distance)


# Category 5: Admission Patterns
def calculate_trend(series, days=7):
    """Calculate trend slope"""
    if len(series) < 2:
        return 0

    recent = series.tail(min(days, len(series))).values

    if len(recent) < 2:
        return 0

    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    return float(slope)


def calculate_icu_ratio(admissions_df):
    """Calculate ICU admission ratio"""
    if admissions_df.empty:
        return 0

    total_icu = admissions_df['icu_admissions'].sum()
    total_admissions = admissions_df['total_admissions'].sum()

    if total_admissions == 0:
        return 0

    return total_icu / total_admissions


def calculate_spike(series):
    """Calculate spike above baseline"""
    if len(series) < 7:
        return 0

    recent = series.tail(7).mean()
    baseline = series.mean()

    if baseline == 0:
        return 0

    return (recent - baseline) / baseline


# Category 6: Resource-Specific
def get_resource_criticality(resource_type):
    """Return predefined criticality score"""
    criticality = {
        'ventilators': 1.0,
        'o2_cylinders': 0.9,
        'medications': 0.8,
        'beds': 0.7,
        'ppe': 0.6
    }
    return criticality.get(resource_type, 0.5)


def calculate_consumption_per_admission(demand_df, admissions_df):
    """Calculate resource consumption per patient admission"""
    if demand_df.empty or admissions_df.empty:
        return 0

    # Use first 7 days
    recent_demand = demand_df['predicted_consumption'].iloc[:7].sum() if len(demand_df) >= 7 else demand_df['predicted_consumption'].sum()
    recent_admissions = admissions_df['total_admissions'].tail(7).sum() if len(admissions_df) >= 7 else admissions_df['total_admissions'].sum()

    if recent_admissions == 0:
        return 0

    return recent_demand / recent_admissions


# Feature names for reference
FEATURE_NAMES = [
    # Category 1: Stock-Demand Ratios
    'stock_demand_ratio',
    'stock_demand_ratio_14d',
    'stock_capacity_ratio',
    'demand_capacity_ratio',

    # Category 2: Time-Based Indicators
    'days_of_supply',
    'days_since_resupply',
    'days_to_critical',
    'predicted_stockout_day',

    # Category 3: Consumption Velocity
    'consumption_trend_7d',
    'consumption_volatility',
    'consumption_acceleration',
    'predicted_demand_change',

    # Category 4: Regional Context
    'regional_avg_stock',
    'regional_transfer_availability',
    'isolation_score',

    # Category 5: Admission Patterns
    'admission_trend_7d',
    'icu_admission_ratio',
    'emergency_admission_spike',

    # Category 6: Resource-Specific
    'resource_criticality',
    'consumption_per_admission'
]
