"""
Feature engineering for shortage detection and preference learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def calculate_rolling_stats(
    df: pd.DataFrame,
    value_col: str,
    windows: List[int] = [7, 14, 30]
) -> pd.DataFrame:
    """Calculate rolling statistics"""
    df = df.copy()
    
    for window in windows:
        df[f'{value_col}_rolling_mean_{window}d'] = df[value_col].rolling(window, min_periods=1).mean()
        df[f'{value_col}_rolling_std_{window}d'] = df[value_col].rolling(window, min_periods=1).std()
        df[f'{value_col}_rolling_min_{window}d'] = df[value_col].rolling(window, min_periods=1).min()
        df[f'{value_col}_rolling_max_{window}d'] = df[value_col].rolling(window, min_periods=1).max()
    
    return df


def calculate_trend(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate linear trend over window"""
    def fit_trend(x):
        if len(x) < 2:
            return 0
        indices = np.arange(len(x))
        slope = np.polyfit(indices, x, 1)[0]
        return slope
    
    return series.rolling(window, min_periods=2).apply(fit_trend, raw=True)


def create_lag_features(
    df: pd.DataFrame,
    value_col: str,
    lags: List[int] = [1, 7, 14, 30]
) -> pd.DataFrame:
    """Create lagged features"""
    df = df.copy()
    
    for lag in lags:
        df[f'{value_col}_lag_{lag}d'] = df[value_col].shift(lag)
    
    return df


def engineer_shortage_features(
    current_inventory: pd.DataFrame,
    demand_predictions: pd.DataFrame,
    admissions_history: pd.DataFrame,
    hospital_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Engineer features for shortage detection model
    
    Args:
        current_inventory: Current stock levels
        demand_predictions: Forecasted demand
        admissions_history: Historical admission data
        hospital_info: Hospital metadata
    
    Returns:
        DataFrame with engineered features
    """
    features = current_inventory.copy()
    
    # Basic inventory features
    features['stock_level'] = features['quantity']
    features['days_of_supply'] = features['quantity'] / (demand_predictions['predicted_demand'].clip(lower=1))
    
    # Demand features
    features['predicted_demand_7d'] = demand_predictions['predicted_demand_7d']
    features['predicted_demand_14d'] = demand_predictions['predicted_demand_14d']
    features['demand_trend'] = demand_predictions['demand_trend']
    
    # Consumption velocity (recent usage rate)
    recent_consumption = admissions_history.groupby('hospital_id').agg({
        'total_admissions': ['mean', 'std', 'max']
    }).reset_index()
    recent_consumption.columns = ['hospital_id', 'avg_admissions', 'std_admissions', 'max_admissions']
    features = features.merge(recent_consumption, on='hospital_id', how='left')
    
    # Hospital capacity features
    # Handle both 'id' and 'hospital_id' column names
    hospital_cols = ['capacity_beds', 'region']
    if 'id' in hospital_info.columns:
        hospital_cols = ['id'] + hospital_cols
        merge_key = 'id'
    elif 'hospital_id' in hospital_info.columns:
        merge_key = 'hospital_id'
    else:
        # If neither exists, try to use the first column as merge key
        merge_key = hospital_info.columns[0] if len(hospital_info.columns) > 0 else None
        if merge_key:
            hospital_cols = [merge_key] + hospital_cols
    
    if merge_key:
        features = features.merge(
            hospital_info[hospital_cols],
            left_on='hospital_id',
            right_on=merge_key,
            how='left'
        )
    features['capacity_utilization'] = features['avg_admissions'] / features['capacity_beds']
    
    # Stock/demand ratio
    features['stock_demand_ratio'] = features['stock_level'] / features['predicted_demand_7d'].clip(lower=1)
    
    # Criticality score
    features['is_below_critical'] = (features['stock_level'] <= features['critical_threshold']).astype(int)
    features['shortage_risk_score'] = (
        (1 - features['days_of_supply'] / 30).clip(0, 1) * 0.5 +
        (1 - features['stock_demand_ratio']).clip(0, 1) * 0.3 +
        features['capacity_utilization'] * 0.2
    )
    
    # Regional context
    regional_avg = features.groupby('region')['stock_level'].transform('mean')
    features['stock_vs_regional_avg'] = features['stock_level'] / regional_avg.clip(lower=1)
    
    # Time features (if date available)
    if 'date' in features.columns:
        features = create_time_features(features)
    
    return features


def extract_recommendation_features(recommendation: Dict) -> Dict:
    """
    Extract features from a recommendation for preference learning
    
    Args:
        recommendation: Dict with allocation strategy details
    
    Returns:
        Dict of numerical features
    """
    features = {
        # Cost features
        'total_cost': recommendation.get('total_cost', 0),
        'cost_per_unit': recommendation.get('cost_per_unit', 0),
        'cost_score': recommendation.get('cost_score', 0.5),
        
        # Speed features
        'estimated_time_hours': recommendation.get('estimated_time_hours', 24),
        'transfer_complexity': recommendation.get('num_transfers', 1),
        'speed_score': recommendation.get('speed_score', 0.5),
        
        # Coverage features
        'hospitals_helped': recommendation.get('hospitals_helped', 1),
        'total_resources_allocated': recommendation.get('total_quantity', 0),
        'shortage_reduction': recommendation.get('shortage_reduction', 0),
        'coverage_score': recommendation.get('coverage_score', 0.5),
        
        # Fairness features
        'regional_balance': recommendation.get('regional_balance', 0.5),
        'priority_alignment': recommendation.get('priority_alignment', 0.5),
        'fairness_score': recommendation.get('fairness_score', 0.5),
        
        # Risk features
        'risk_level_addressed': recommendation.get('risk_level', 2),  # 0-3 scale
        'urgency_level': recommendation.get('urgency', 2),
        
        # Logistics
        'num_source_hospitals': recommendation.get('num_sources', 1),
        'average_transfer_distance': recommendation.get('avg_distance_km', 50)
    }
    
    return features