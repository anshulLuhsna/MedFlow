#!/usr/bin/env python3
"""
Real-World Prediction Scenarios Test

Tests the trained models on realistic hospital scenarios:
1. Normal operations
2. Flu season surge
3. Weekend low activity
4. Emergency situation
5. Post-holiday spike

Shows how to use the models for actual forecasting.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add ml_core to path
ml_core_dir = Path(__file__).parent / 'ml_core'
sys.path.insert(0, str(ml_core_dir))
sys.path.insert(0, str(Path(__file__).parent))

from ml_core.models.demand_forecaster import DemandForecaster
from ml_core.models.calibrated_forecaster import CalibratedPPEForecaster
from ml_core.utils.demo_data import create_demo_features


class HospitalScenario:
    """Generate realistic hospital admission patterns"""

    @staticmethod
    def _generate_from_admissions(admissions, seed):
        """
        Generate all features from admission patterns

        Args:
            admissions: Array of daily admissions
            seed: Random seed

        Returns:
            (days, 17) feature array
        """
        np.random.seed(seed)
        days = len(admissions)

        # Generate consumption based on admissions (with noise)
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

            quantity[i] = max(0, new_qty)

        # Create all engineered features
        features = create_demo_features(quantity, consumption, resupply, admissions)

        return features

    @staticmethod
    def normal_operations(days=30):
        """Normal hospital operations - baseline scenario"""
        np.random.seed(42)
        admissions = []

        for day in range(days):
            day_of_week = day % 7

            # Base admissions with weekly pattern
            base_admissions = 45
            if day_of_week == 0:  # Monday surge
                total = base_admissions + 10
            elif day_of_week in [5, 6]:  # Weekend dip
                total = base_admissions - 15
            else:
                total = base_admissions

            # Add some randomness
            total += np.random.randint(-5, 6)
            total = max(10, total)  # Minimum 10 admissions

            admissions.append(total)

        return HospitalScenario._generate_from_admissions(np.array(admissions), 42)

    @staticmethod
    def flu_season_surge(days=30):
        """Flu season - increased admissions, especially emergency"""
        np.random.seed(43)
        admissions_list = []

        for day in range(days):
            day_of_week = day % 7

            # Higher base admissions
            base_admissions = 65
            if day_of_week == 0:
                total = base_admissions + 15
            elif day_of_week in [5, 6]:
                total = base_admissions - 10
            else:
                total = base_admissions

            total += np.random.randint(-8, 12)
            total = max(20, total)

            admissions_list.append(total)

        return HospitalScenario._generate_from_admissions(np.array(admissions_list), 43)

    @staticmethod
    def weekend_low_activity(days=30):
        """Weekend/holiday period - reduced scheduled admissions"""
        np.random.seed(44)
        admissions_list = []

        for day in range(days):
            day_of_week = day % 7

            # Lower scheduled admissions
            base_admissions = 30
            if day_of_week in [5, 6]:  # Weekends very low
                total = base_admissions - 10
            else:
                total = base_admissions

            total += np.random.randint(-5, 6)
            total = max(10, total)

            admissions_list.append(total)

        return HospitalScenario._generate_from_admissions(np.array(admissions_list), 44)

    @staticmethod
    def emergency_situation(days=30):
        """Major incident - surge in emergency admissions"""
        np.random.seed(45)
        admissions_list = []

        for day in range(days):
            # Spike in middle of period (days 10-15)
            if 10 <= day <= 15:
                base_admissions = 90  # Major surge
            else:
                base_admissions = 45

            total = base_admissions + np.random.randint(-10, 11)
            total = max(15, total)

            admissions_list.append(total)

        return HospitalScenario._generate_from_admissions(np.array(admissions_list), 45)

    @staticmethod
    def post_holiday_spike(days=30):
        """Post-holiday - backlog of scheduled procedures"""
        np.random.seed(46)
        admissions_list = []

        for day in range(days):
            day_of_week = day % 7

            # Higher scheduled admissions (catching up)
            base_admissions = 55
            if day_of_week == 0:  # Monday rush
                total = base_admissions + 20
            elif day_of_week in [5, 6]:
                total = base_admissions - 5
            else:
                total = base_admissions + 5

            total += np.random.randint(-7, 8)
            total = max(15, total)

            admissions_list.append(total)

        return HospitalScenario._generate_from_admissions(np.array(admissions_list), 46)


def predict_scenario(forecaster, scenario_name, scenario_data, use_calibrated=True):
    """
    Make predictions for a scenario

    Args:
        forecaster: Loaded forecaster model
        scenario_name: Name of scenario
        scenario_data: 30 days of historical data (30, 17)
        use_calibrated: Whether using calibrated forecaster
    """
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}\n")

    # Show last 3 days of input
    # Feature columns: [quantity, consumption, resupply, admissions, ...]
    print("Recent trends (last 3 days):")
    print(f"  {'Day':<8} {'Admissions':<12} {'Consumption':<12} {'Inventory':<12}")
    print(f"  {'-'*50}")
    for i in range(-3, 0):
        day = scenario_data[i]
        # day[3]=admissions, day[1]=consumption, day[0]=quantity
        print(f"  Day {30+i:<3} {day[3]:<12.1f} {day[1]:<12.1f} {day[0]:<12.1f}")

    # Prepare input
    X = scenario_data.reshape(1, 30, 17)

    # Point prediction
    print(f"\n{'Point Predictions (Next 14 Days)':-^80}")
    point_pred = forecaster.predict(X, probabilistic=False)

    print(f"\nNext week forecast (days 1-7):")
    print(f"  Day 1: {point_pred[0, 0]:.1f} sets")
    print(f"  Day 2: {point_pred[0, 1]:.1f} sets")
    print(f"  Day 3: {point_pred[0, 2]:.1f} sets")
    print(f"  Day 7: {point_pred[0, 6]:.1f} sets")
    print(f"  Average: {point_pred[0, :7].mean():.1f} sets/day")

    print(f"\nSecond week forecast (days 8-14):")
    print(f"  Average: {point_pred[0, 7:14].mean():.1f} sets/day")

    # Probabilistic prediction
    print(f"\n{'Probabilistic Predictions (with Uncertainty)':-^80}")
    print("Running MC Dropout (this may take 1-2 minutes)...")

    prob_pred = forecaster.predict(X, probabilistic=True, n_samples=200)

    # Tomorrow's detailed forecast
    mean = prob_pred['mean'][0, 0]
    p10 = prob_pred['p10'][0, 0]
    p50 = prob_pred['p50'][0, 0]
    p90 = prob_pred['p90'][0, 0]
    std = prob_pred['std'][0, 0]

    print(f"\nTomorrow's demand (Day 1):")
    print(f"  Expected (mean):        {mean:>6.1f} sets")
    print(f"  Median (P50):           {p50:>6.1f} sets")
    print(f"  Optimistic (P10):       {p10:>6.1f} sets")
    print(f"  Conservative (P90):     {p90:>6.1f} sets")
    print(f"  Uncertainty (±std):     {std:>6.1f} sets")
    print(f"  80% Confidence Range:   [{p10:.1f}, {p90:.1f}]")

    # Ordering recommendation
    SAFETY_BUFFER = 1.15 if use_calibrated else 1.25
    recommended_order = p90 * SAFETY_BUFFER

    print(f"\n{'Ordering Recommendation':-^80}")
    print(f"  Base forecast (P90):           {p90:>6.1f} sets")
    print(f"  Safety buffer ({(SAFETY_BUFFER-1)*100:.0f}%):            {p90*(SAFETY_BUFFER-1):>6.1f} sets")
    print(f"  Recommended order quantity:    {recommended_order:>6.0f} sets")

    if recommended_order < 10:
        risk = "LOW"
    elif recommended_order < 20:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    print(f"  Risk level:                    {risk}")

    # Weekly summary
    print(f"\n{'Weekly Planning Summary':-^80}")
    weekly_mean = prob_pred['mean'][0, :7].mean()
    weekly_p90 = prob_pred['p90'][0, :7].mean()
    weekly_order = weekly_p90 * 7 * SAFETY_BUFFER

    print(f"  Average daily demand (next 7 days):  {weekly_mean:.1f} sets/day")
    print(f"  Conservative daily (P90):             {weekly_p90:.1f} sets/day")
    print(f"  Weekly order quantity:                {weekly_order:.0f} sets total")

    return {
        'scenario': scenario_name,
        'tomorrow_mean': mean,
        'tomorrow_p90': p90,
        'recommended_order': recommended_order,
        'weekly_total': weekly_order,
        'risk_level': risk
    }


def main():
    """Run all scenarios"""
    print(f"\n{'='*80}")
    print("REAL-WORLD DEMAND FORECASTING TEST")
    print("Medical Resource Allocation System - PPE Predictions")
    print(f"{'='*80}\n")

    # Load model (calibrated version recommended)
    print("Loading calibrated PPE forecaster...")
    try:
        forecaster = CalibratedPPEForecaster()
        forecaster.load()
        use_calibrated = True
        print("✓ Calibrated forecaster loaded (includes automatic calibration adjustment)")
    except Exception as e:
        print(f"Could not load calibrated forecaster: {e}")
        print("Falling back to standard forecaster...")
        forecaster = DemandForecaster('ppe')
        forecaster.load()
        use_calibrated = False
        print("✓ Standard forecaster loaded (manual safety buffer recommended)")

    # Generate scenarios
    scenarios = {
        "1. Normal Operations": HospitalScenario.normal_operations(),
        "2. Flu Season Surge": HospitalScenario.flu_season_surge(),
        "3. Weekend/Holiday Low Activity": HospitalScenario.weekend_low_activity(),
        "4. Emergency Situation": HospitalScenario.emergency_situation(),
        "5. Post-Holiday Backlog": HospitalScenario.post_holiday_spike()
    }

    # Run predictions
    results = []
    for name, data in scenarios.items():
        result = predict_scenario(forecaster, name, data, use_calibrated)
        results.append(result)
        input("\nPress Enter to continue to next scenario...")

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: All Scenarios Comparison")
    print(f"{'='*80}\n")

    print(f"{'Scenario':<30} {'Tomorrow':<12} {'P90':<12} {'Order':<12} {'Risk':<10}")
    print(f"{'-'*76}")
    for r in results:
        print(f"{r['scenario']:<30} {r['tomorrow_mean']:>6.1f} sets  {r['tomorrow_p90']:>6.1f} sets  "
              f"{r['recommended_order']:>6.0f} sets  {r['risk_level']:<10}")

    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")

    # Find extremes
    max_demand = max(results, key=lambda x: x['tomorrow_mean'])
    min_demand = min(results, key=lambda x: x['tomorrow_mean'])

    print(f"Highest demand scenario:  {max_demand['scenario']}")
    print(f"  Tomorrow: {max_demand['tomorrow_mean']:.1f} sets (order {max_demand['recommended_order']:.0f})")
    print()
    print(f"Lowest demand scenario:   {min_demand['scenario']}")
    print(f"  Tomorrow: {min_demand['tomorrow_mean']:.1f} sets (order {min_demand['recommended_order']:.0f})")
    print()
    print(f"Demand variation: {max_demand['tomorrow_mean'] - min_demand['tomorrow_mean']:.1f} sets "
          f"({(max_demand['tomorrow_mean'] / min_demand['tomorrow_mean'] - 1) * 100:.0f}% difference)")

    print(f"\n{'='*80}")
    print("PRODUCTION DEPLOYMENT NOTES")
    print(f"{'='*80}\n")

    print("1. Use CalibratedPPEForecaster for production predictions")
    print("2. Apply 15% safety buffer to P90 for conservative ordering")
    print("3. Monitor actual demand vs. predictions weekly")
    print("4. Retrain models monthly with real data")
    print("5. Set alerts when P90 demand exceeds inventory thresholds")
    print()
    print("Model Performance:")
    print(f"  - Point prediction MAE: 4.65 sets")
    print(f"  - Probabilistic coverage: ~75-80% (with calibration + buffer)")
    print(f"  - Suitable for: Daily ordering, weekly planning, risk assessment")

    print(f"\n{'='*80}")
    print("Test Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
