#!/usr/bin/env python3
"""
Quick Prediction Demo

Shows how to make a single prediction with the trained models.
Perfect for testing and understanding the API.
"""

import sys
import numpy as np
from pathlib import Path

# Add ml_core to path
ml_core_dir = Path(__file__).parent / 'ml_core'
sys.path.insert(0, str(ml_core_dir))
sys.path.insert(0, str(Path(__file__).parent))

from ml_core.models.calibrated_forecaster import CalibratedPPEForecaster
from ml_core.models.demand_forecaster import DemandForecaster
from ml_core.utils.demo_data import create_sample_hospital_data


def create_sample_input():
    """
    Create sample input data representing last 30 days with all 17 features

    Returns:
        X: (1, 30, 17) array with engineered features
    """
    print("Creating sample hospital data for last 30 days...")
    print("=" * 60)

    # Generate realistic hospital data with proper feature engineering
    data = create_sample_hospital_data(days=40, seed=42)  # Generate 40 days, use last 30

    # Extract last 30 days
    features_30d = data['last_30']

    # Show last 5 days summary
    print("\nLast 5 days summary:")
    print(f"{'Day':<6} {'Admissions':<12} {'Consumption':<12} {'Inventory':<12}")
    print("-" * 50)

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i in range(-5, 0):
        day_idx = (30 + i) % 7
        row = features_30d[i]
        print(f"{day_names[day_idx]:<6} {row[3]:<12.1f} {row[1]:<12.1f} {row[0]:<12.1f}")

    print(f"\nRecent trends:")
    print(f"  7-day avg inventory:  {features_30d[-1, 4]:.1f}")
    print(f"  14-day avg inventory: {features_30d[-1, 5]:.1f}")
    print(f"  Inventory trend:      {features_30d[-1, 6]:+.2f} (slope)")

    return features_30d.reshape(1, 30, 17)


def make_prediction_ppe():
    """Make PPE prediction using calibrated forecaster"""
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    # Load calibrated forecaster
    forecaster = CalibratedPPEForecaster()
    forecaster.load()
    print("✓ Calibrated PPE forecaster loaded")

    # Create sample input
    X = create_sample_input()

    # Point prediction (fast)
    print("\n" + "=" * 60)
    print("POINT PREDICTION (Fast)")
    print("=" * 60)

    point_pred = forecaster.predict(X, probabilistic=False)

    print(f"\n14-Day Forecast:")
    print(f"  Tomorrow (Day 1):     {point_pred[0, 0]:>6.1f} sets")
    print(f"  Day 2:                {point_pred[0, 1]:>6.1f} sets")
    print(f"  Day 3:                {point_pred[0, 2]:>6.1f} sets")
    print(f"  Day 7:                {point_pred[0, 6]:>6.1f} sets")
    print(f"  Day 14:               {point_pred[0, 13]:>6.1f} sets")
    print(f"\n  Next week average:    {point_pred[0, :7].mean():>6.1f} sets/day")
    print(f"  Second week average:  {point_pred[0, 7:14].mean():>6.1f} sets/day")

    # Probabilistic prediction (slower but with uncertainty)
    print("\n" + "=" * 60)
    print("PROBABILISTIC PREDICTION (Slower, includes uncertainty)")
    print("=" * 60)
    print("Running MC Dropout with 200 samples (takes ~1 minute)...\n")

    prob_pred = forecaster.predict(X, probabilistic=True, n_samples=200)

    # Tomorrow's detailed forecast
    print("\nTOMORROW'S FORECAST (Day 1):")
    print("=" * 60)

    mean = prob_pred['mean'][0, 0]
    p10 = prob_pred['p10'][0, 0]
    p50 = prob_pred['p50'][0, 0]
    p90 = prob_pred['p90'][0, 0]
    std = prob_pred['std'][0, 0]

    print(f"\nDemand Estimates:")
    print(f"  Expected (mean):           {mean:>6.1f} sets")
    print(f"  Median (P50):              {p50:>6.1f} sets")
    print(f"  Uncertainty (±std):        {std:>6.1f} sets")

    print(f"\nConfidence Intervals:")
    print(f"  Optimistic (P10):          {p10:>6.1f} sets  (10% chance of being lower)")
    print(f"  Conservative (P90):        {p90:>6.1f} sets  (90% chance of being lower)")
    print(f"  80% Confidence Range:      [{p10:.1f}, {p90:.1f}]")

    # Ordering recommendation
    print("\n" + "=" * 60)
    print("ORDERING RECOMMENDATION")
    print("=" * 60)

    SAFETY_BUFFER = 1.15  # 15% safety margin
    recommended = p90 * SAFETY_BUFFER

    print(f"\nFor tomorrow:")
    print(f"  Expected demand:           {mean:>6.1f} sets")
    print(f"  Conservative (P90):        {p90:>6.1f} sets")
    print(f"  Safety buffer (15%):       {p90 * 0.15:>6.1f} sets")
    print(f"  ───────────────────────────────────")
    print(f"  RECOMMENDED ORDER:         {recommended:>6.0f} sets")

    if recommended < 10:
        print(f"  Risk level: LOW")
        print(f"  Note: Minimal demand expected")
    elif recommended < 20:
        print(f"  Risk level: MODERATE")
        print(f"  Note: Normal operations expected")
    else:
        print(f"  Risk level: HIGH")
        print(f"  Note: Elevated demand expected, ensure adequate stock")

    # Weekly planning
    weekly_mean = prob_pred['mean'][0, :7].mean()
    weekly_p90 = prob_pred['p90'][0, :7].mean()
    weekly_total = weekly_p90 * 7 * SAFETY_BUFFER

    print(f"\nWeekly Planning (Next 7 days):")
    print(f"  Average daily demand:      {weekly_mean:>6.1f} sets/day")
    print(f"  Conservative daily:        {weekly_p90:>6.1f} sets/day")
    print(f"  Total weekly order:        {weekly_total:>6.0f} sets")

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
What these numbers mean:

1. Expected (mean): Most likely demand based on historical patterns
   → Use for budget planning and average inventory levels

2. P90 (Conservative): 90% chance actual demand will be lower
   → Use for ordering decisions to avoid stockouts
   → Only 10% chance you'll need more than this

3. Recommended Order: P90 + 15% safety buffer
   → Accounts for model uncertainty and unexpected surges
   → ~75-80% confidence of having enough stock
   → Industry standard for critical medical supplies

4. 80% Confidence Range: Where actual demand will fall 80% of the time
   → Helps assess risk and variability
   → Wider range = more uncertainty

Example Decision Tree:
- If P90 < 10 sets: Order 10 (minimum stock)
- If P90 10-15 sets: Order P90 × 1.15 (standard buffer)
- If P90 > 15 sets: Order P90 × 1.20 (extra buffer for high demand)
    """)

    print("=" * 60)
    print("DONE! Model is ready for production use.")
    print("=" * 60)


def make_prediction_other_resource(resource_type):
    """Make prediction for other resources (O2, ventilators, etc.)"""
    print("\n" + "=" * 60)
    print(f"PREDICTION FOR: {resource_type.upper()}")
    print("=" * 60)

    forecaster = DemandForecaster(resource_type)
    forecaster.load()
    print(f"✓ {resource_type} forecaster loaded")

    # Create sample input
    X = create_sample_input()

    # Point prediction
    point_pred = forecaster.predict(X, probabilistic=False)

    print(f"\n14-Day Forecast:")
    print(f"  Tomorrow:     {point_pred[0, 0]:.2f} units")
    print(f"  Week average: {point_pred[0, :7].mean():.2f} units/day")

    print("\nNote: For calibrated probabilistic predictions,")
    print("run apply_calibration_fix.py to create calibrated wrapper.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Quick prediction demo')
    parser.add_argument(
        '--resource',
        type=str,
        default='ppe',
        choices=['ppe', 'o2_cylinders', 'ventilators', 'medications', 'beds'],
        help='Resource type to predict'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MEDFLOW DEMAND FORECASTING - QUICK DEMO")
    print("=" * 60)

    if args.resource == 'ppe':
        make_prediction_ppe()
    else:
        make_prediction_other_resource(args.resource)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Test realistic scenarios:
   python3 test_real_world_scenarios.py

2. Integrate with your database:
   - Load last 30 days from DB
   - Make prediction
   - Save forecast to DB

3. Deploy as API:
   - See LSTM_DEMAND_FORECASTING_GUIDE.md
   - REST API example included

4. Set up monitoring:
   - Track actual vs predicted
   - Retrain monthly with real data
    """)
