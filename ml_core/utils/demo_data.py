"""
Helper functions to create properly formatted demo data for predictions

The models expect 17 engineered features, not just raw admission data.
This module helps create correctly formatted input data.
"""

import numpy as np
import pandas as pd


def create_demo_features(
    quantity_history,
    consumption_history,
    resupply_history,
    admissions_history
):
    """
    Create all 17 required features from basic history

    Args:
        quantity_history: List/array of inventory quantities (length 30+)
        consumption_history: List/array of consumption values (length 30+)
        resupply_history: List/array of resupply amounts (length 30+)
        admissions_history: List/array of total admissions (length 30+)

    Returns:
        np.array of shape (len, 17) with all engineered features
    """
    # Convert to numpy
    quantity = np.array(quantity_history, dtype=float)
    consumption = np.array(consumption_history, dtype=float)
    resupply = np.array(resupply_history, dtype=float)
    admissions = np.array(admissions_history, dtype=float)

    n = len(quantity)

    # Initialize feature array
    features = np.zeros((n, 17))

    # Base features (4)
    features[:, 0] = quantity  # quantity
    features[:, 1] = consumption  # consumption
    features[:, 2] = resupply  # resupply
    features[:, 3] = admissions  # total_admissions

    # Calculate rolling means
    quantity_ma_7d = pd.Series(quantity).rolling(window=7, min_periods=1).mean().values
    quantity_ma_14d = pd.Series(quantity).rolling(window=14, min_periods=1).mean().values

    features[:, 4] = quantity_ma_7d  # quantity_ma_7d
    features[:, 5] = quantity_ma_14d  # quantity_ma_14d

    # Calculate trends (slope over last 14 days)
    quantity_trend = calculate_trend(quantity, window=14)
    consumption_trend = calculate_trend(consumption, window=14)

    features[:, 6] = quantity_trend  # quantity_trend
    features[:, 7] = consumption_trend  # consumption_trend

    # Change features
    quantity_change = np.diff(quantity, prepend=quantity[0])
    consumption_change = np.diff(consumption, prepend=consumption[0])

    features[:, 8] = quantity_change  # quantity_change
    features[:, 9] = consumption_change  # consumption_change

    # Normalized features
    quantity_per_admission = quantity / (admissions + 1)
    consumption_rate = consumption / (admissions + 1)

    features[:, 10] = quantity_per_admission  # quantity_per_admission
    features[:, 11] = consumption_rate  # consumption_rate

    # Momentum features (acceleration)
    quantity_momentum = np.diff(quantity_change, prepend=quantity_change[0])
    consumption_momentum = np.diff(consumption_change, prepend=consumption_change[0])

    features[:, 12] = quantity_momentum  # quantity_momentum
    features[:, 13] = consumption_momentum  # consumption_momentum

    # Percentage changes
    quantity_pct = pd.Series(quantity).pct_change().fillna(0).replace([np.inf, -np.inf], 0).values
    consumption_pct = pd.Series(consumption).pct_change().fillna(0).replace([np.inf, -np.inf], 0).values

    features[:, 14] = quantity_pct  # quantity_pct_change
    features[:, 15] = consumption_pct  # consumption_pct_change

    # Trend direction
    trend_direction = np.sign(quantity_trend)

    features[:, 16] = trend_direction  # trend_direction

    return features


def calculate_trend(values, window=14):
    """Calculate slope/trend over a rolling window"""
    n = len(values)
    trends = np.zeros(n)

    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_values = values[start_idx:i+1]

        if len(window_values) < 2:
            trends[i] = 0
            continue

        # Simple linear regression slope
        x = np.arange(len(window_values))
        y = window_values

        # Slope = cov(x,y) / var(x)
        if np.var(x) > 0:
            slope = np.cov(x, y)[0, 1] / np.var(x)
            trends[i] = slope
        else:
            trends[i] = 0

    return trends


def create_sample_hospital_data(days=30, seed=42):
    """
    Create realistic sample hospital data for demo purposes

    Args:
        days: Number of days of history to generate
        seed: Random seed for reproducibility

    Returns:
        Dict with all required histories and properly formatted features
    """
    np.random.seed(seed)

    # Generate realistic admission patterns
    admissions = []
    for day in range(days):
        day_of_week = day % 7
        if day_of_week in [5, 6]:  # Weekend
            base = 30
        else:  # Weekday
            base = 45
        admissions.append(max(10, base + np.random.randint(-7, 8)))

    admissions = np.array(admissions)

    # Generate consumption based on admissions (with some noise)
    consumption = (admissions * 0.25 + np.random.randn(days) * 1.5).clip(0)

    # Generate inventory with resupply logic
    quantity = np.zeros(days)
    resupply = np.zeros(days)

    quantity[0] = 120  # Starting inventory

    for i in range(1, days):
        # Inventory after consumption
        new_qty = quantity[i-1] - consumption[i-1]

        # Resupply if below threshold
        if new_qty < 50:
            resupply_amount = np.random.randint(50, 100)
            resupply[i] = resupply_amount
            new_qty += resupply_amount

        quantity[i] = new_qty

    # Create all engineered features
    features = create_demo_features(quantity, consumption, resupply, admissions)

    return {
        'features': features,  # (days, 17) array
        'quantity': quantity,
        'consumption': consumption,
        'resupply': resupply,
        'admissions': admissions,
        'last_30': features[-30:] if len(features) >= 30 else features  # Last 30 days for prediction
    }


# Feature column names for reference
FEATURE_NAMES = [
    'quantity',                 # 0
    'consumption',              # 1
    'resupply',                 # 2
    'total_admissions',         # 3
    'quantity_ma_7d',           # 4
    'quantity_ma_14d',          # 5
    'quantity_trend',           # 6
    'consumption_trend',        # 7
    'quantity_change',          # 8
    'consumption_change',       # 9
    'quantity_per_admission',   # 10
    'consumption_rate',         # 11
    'quantity_momentum',        # 12
    'consumption_momentum',     # 13
    'quantity_pct_change',      # 14
    'consumption_pct_change',   # 15
    'trend_direction'           # 16
]
