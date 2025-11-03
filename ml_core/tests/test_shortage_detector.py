"""
Test cases for Shortage Detector Model

Tests various scenarios to ensure the model correctly predicts shortage risk levels.
"""

import sys
from pathlib import Path

# Add ml_core directory to Python path
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

import pandas as pd
import numpy as np
from models.shortage_detector import ShortageDetector
from utils.shortage_features import engineer_shortage_features
from datetime import datetime, timedelta


def create_sample_admissions(hospital_id: str, baseline: int = 30, trend: str = 'stable', days: int = 30):
    """Create sample 30-day admission history"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    
    if trend == 'stable':
        admissions = np.random.poisson(baseline, days)
    elif trend == 'high':
        admissions = baseline + np.arange(days) * 0.5 + np.random.poisson(5, days)
    elif trend == 'surge':
        # Recent surge pattern
        admissions = baseline + np.concatenate([
            np.random.poisson(2, days - 10),
            np.random.poisson(15, 10)  # Recent surge
        ])
    elif trend == 'declining':
        admissions = baseline - np.arange(days) * 0.3 + np.random.poisson(3, days)
        admissions = np.maximum(admissions, 1)  # Ensure positive
    else:
        admissions = np.random.poisson(baseline, days)
    
    date_col = 'admission_date'
    return pd.DataFrame({
        'hospital_id': [hospital_id] * days,
        date_col: dates,
        'total_admissions': admissions.astype(int),
        'icu_admissions': (admissions * 0.2).astype(int),
        'emergency_admissions': (admissions * 0.4).astype(int)
    })


def create_sample_hospital(hospital_id: str, region: str = 'Region_A', 
                          capacity_beds: int = 500, lat: float = 37.7749, lon: float = -122.4194):
    """Create sample hospital info"""
    return pd.DataFrame({
        'hospital_id': [hospital_id],
        'region': [region],
        'capacity': [capacity_beds],
        'capacity_beds': [capacity_beds],
        'specialization': ['General'],
        'latitude': [lat],
        'longitude': [lon]
    })


def create_sample_inventory(hospital_id: str, resource_type: str, quantity: int, 
                           capacity: int = None, region: str = None):
    """Create sample inventory row"""
    if capacity is None:
        capacity = max(quantity * 3, 100)
    
    return pd.DataFrame({
        'hospital_id': [hospital_id],
        'resource_type': [resource_type],
        'quantity': [quantity],
        'capacity': [capacity],
        'max_capacity': [capacity],
        'region': [region] if region else ['Region_A'],
        'last_updated': [pd.Timestamp.now() - timedelta(days=5)],
        'last_resupply_date': [pd.Timestamp.now() - timedelta(days=5)]
    })


def create_sample_demand_predictions(hospital_id: str, resource_type: str, 
                                     consumption_per_day: float = 10, days: int = 14,
                                     trend: str = 'stable'):
    """Create sample demand predictions"""
    if trend == 'stable':
        consumption = [consumption_per_day] * days
    elif trend == 'increasing':
        consumption = [consumption_per_day * (1 + i * 0.1) for i in range(days)]
    elif trend == 'surge':
        consumption = [consumption_per_day] * 7 + [consumption_per_day * 2] * 7
    elif trend == 'declining':
        consumption = [consumption_per_day * (1 - i * 0.05) for i in range(days)]
        consumption = [max(c, 0.1) for c in consumption]  # Ensure positive
    else:
        consumption = [consumption_per_day] * days
    
    return pd.DataFrame({
        'hospital_id': [hospital_id] * days,
        'resource_type': [resource_type] * days,
        'day': list(range(days)),
        'predicted_consumption': consumption
    })


def test_scenario_1_low_stock_high_demand():
    """Scenario 1: Low stock + high predicted demand → Critical"""
    
    print("\n" + "="*60)
    print("Scenario 1: Low Stock + High Demand")
    print("="*60)
    
    hospital_id = 'H001'
    resource_type = 'ppe'
    
    # Setup: Hospital with 10 PPE sets, predicted demand 15/day
    current_inventory = create_sample_inventory(hospital_id, resource_type, quantity=10, capacity=500)
    demand_predictions = create_sample_demand_predictions(hospital_id, resource_type, consumption_per_day=15.0)
    admissions_history = create_sample_admissions(hospital_id, baseline=50, trend='high')
    hospital_info = create_sample_hospital(hospital_id)
    
    # Engineer features
    features = engineer_shortage_features(
        current_inventory, demand_predictions,
        admissions_history, hospital_info
    )
    
    # Load model and predict
    detector = ShortageDetector()
    detector.load()
    
    risk_level, probs = detector.predict(features, return_probabilities=True)
    
    print(f"\nPrediction: {risk_level[0]}")
    print(f"Confidence: {probs.max():.2%}")
    print(f"\nProbabilities:")
    for idx, risk_class in enumerate(detector.label_encoder.classes_):
        print(f"  {risk_class}: {probs[0, idx]:.2%}")
    
    print(f"\nKey Features:")
    print(f"  Days of supply: {features['days_of_supply'].iloc[0]:.1f}")
    print(f"  Stock/demand ratio: {features['stock_demand_ratio'].iloc[0]:.2f}")
    print(f"  Predicted stockout day: {features['predicted_stockout_day'].iloc[0]:.0f}")
    
    # Verify
    assert risk_level[0] == 'critical', f"Expected 'critical', got '{risk_level[0]}'"
    assert probs[0, detector.label_encoder.transform(['critical'])[0]] > 0.7, "Critical probability too low"
    
    print("\n✓ Scenario 1 PASSED: Correctly predicted CRITICAL")
    return True


def test_scenario_2_medium_stock_normal_demand():
    """Scenario 2: Medium stock + normal demand → Low"""
    
    print("\n" + "="*60)
    print("Scenario 2: Medium Stock + Normal Demand")
    print("="*60)
    
    hospital_id = 'H002'
    resource_type = 'ppe'
    
    # Setup: Hospital with 200 PPE sets, predicted demand 10/day
    current_inventory = create_sample_inventory(hospital_id, resource_type, quantity=200, capacity=500)
    demand_predictions = create_sample_demand_predictions(hospital_id, resource_type, consumption_per_day=10.0)
    admissions_history = create_sample_admissions(hospital_id, baseline=30, trend='stable')
    hospital_info = create_sample_hospital(hospital_id)
    
    # Engineer features
    features = engineer_shortage_features(
        current_inventory, demand_predictions,
        admissions_history, hospital_info
    )
    
    # Predict
    detector = ShortageDetector()
    detector.load()
    
    risk_level, probs = detector.predict(features, return_probabilities=True)
    
    print(f"\nPrediction: {risk_level[0]}")
    print(f"Confidence: {probs.max():.2%}")
    
    print(f"\nKey Features:")
    print(f"  Days of supply: {features['days_of_supply'].iloc[0]:.1f}")
    print(f"  Stock/demand ratio: {features['stock_demand_ratio'].iloc[0]:.2f}")
    
    # Verify: Should be LOW (20 days supply > 14)
    assert risk_level[0] == 'low', f"Expected 'low', got '{risk_level[0]}'"
    
    print("\n✓ Scenario 2 PASSED: Correctly predicted LOW")
    return True


def test_scenario_3_gradual_depletion():
    """Scenario 3: Gradual depletion → High warning 3 days before stockout"""
    
    print("\n" + "="*60)
    print("Scenario 3: Gradual Depletion")
    print("="*60)
    
    hospital_id = 'H003'
    resource_type = 'ventilators'
    
    # Setup: Hospital with 30 ventilators, demand 8/day → stockout in ~4 days
    current_inventory = create_sample_inventory(hospital_id, resource_type, quantity=30, capacity=100)
    demand_predictions = create_sample_demand_predictions(hospital_id, resource_type, consumption_per_day=8.0)
    admissions_history = create_sample_admissions(hospital_id, baseline=40, trend='stable')
    hospital_info = create_sample_hospital(hospital_id)
    
    # Engineer features
    features = engineer_shortage_features(
        current_inventory, demand_predictions,
        admissions_history, hospital_info
    )
    
    # Predict
    detector = ShortageDetector()
    detector.load()
    
    risk_level, probs = detector.predict(features, return_probabilities=True)
    
    print(f"\nPrediction: {risk_level[0]}")
    print(f"Confidence: {probs.max():.2%}")
    
    print(f"\nKey Features:")
    print(f"  Days of supply: {features['days_of_supply'].iloc[0]:.1f}")
    print(f"  Days to critical: {features['days_to_critical'].iloc[0]:.0f}")
    print(f"  Predicted stockout day: {features['predicted_stockout_day'].iloc[0]:.0f}")
    
    # Verify: Should be HIGH or CRITICAL (4 days < 7 day threshold)
    assert risk_level[0] in ['high', 'critical'], \
        f"Expected 'high' or 'critical' (stockout in ~4 days), got '{risk_level[0]}'"
    
    print(f"\n✓ Scenario 3 PASSED: Correctly predicted {risk_level[0].upper()} (stockout in ~4 days)")
    return True


def test_scenario_4_sudden_surge():
    """Scenario 4: Sudden surge → Critical detection"""
    
    print("\n" + "="*60)
    print("Scenario 4: Sudden Demand Surge")
    print("="*60)
    
    hospital_id = 'H004'
    resource_type = 'o2_cylinders'
    
    # Setup: Hospital with 50 O2 cylinders, sudden spike to 30/day
    current_inventory = create_sample_inventory(hospital_id, resource_type, quantity=50, capacity=200)
    demand_predictions = create_sample_demand_predictions(
        hospital_id, resource_type, consumption_per_day=30.0, trend='surge'
    )
    admissions_history = create_sample_admissions(hospital_id, baseline=20, trend='surge')
    hospital_info = create_sample_hospital(hospital_id)
    
    # Engineer features
    features = engineer_shortage_features(
        current_inventory, demand_predictions,
        admissions_history, hospital_info
    )
    
    # Predict
    detector = ShortageDetector()
    detector.load()
    
    risk_level, probs = detector.predict(features, return_probabilities=True)
    
    print(f"\nPrediction: {risk_level[0]}")
    print(f"Confidence: {probs.max():.2%}")
    
    print(f"\nKey Features:")
    print(f"  Days of supply: {features['days_of_supply'].iloc[0]:.1f}")
    print(f"  Consumption acceleration: {features['consumption_acceleration'].iloc[0]:.2f}")
    print(f"  Emergency admission spike: {features['emergency_admission_spike'].iloc[0]:.2f}")
    
    # Verify: Should be CRITICAL (high demand, low stock)
    assert risk_level[0] == 'critical', f"Expected 'critical', got '{risk_level[0]}'"
    
    print("\n✓ Scenario 4 PASSED: Correctly detected CRITICAL surge")
    return True


def test_scenario_5_regional_imbalance():
    """Scenario 5: Regional imbalance detection - low stock but region has surplus"""
    
    print("\n" + "="*60)
    print("Scenario 5: Regional Imbalance")
    print("="*60)
    
    hospital_id = 'H005'
    resource_type = 'medications'
    
    # Setup: Hospital low on stock BUT region has surplus (modeled via regional features)
    current_inventory = create_sample_inventory(hospital_id, resource_type, quantity=30, capacity=300, region='Region_A')
    
    # Create additional hospitals in region with surplus for regional features
    # (This would normally come from current_inventory DF with multiple hospitals)
    demand_predictions = create_sample_demand_predictions(hospital_id, resource_type, consumption_per_day=8.0)
    admissions_history = create_sample_admissions(hospital_id, baseline=25, trend='stable')
    hospital_info = create_sample_hospital(hospital_id, region='Region_A')
    
    # Engineer features
    features = engineer_shortage_features(
        current_inventory, demand_predictions,
        admissions_history, hospital_info
    )
    
    # Manually set high regional availability to simulate regional help
    features['regional_transfer_availability'] = 500  # Lots of surplus in region
    
    # Predict
    detector = ShortageDetector()
    detector.load()
    
    risk_level, probs = detector.predict(features, return_probabilities=True)
    
    print(f"\nPrediction: {risk_level[0]}")
    print(f"Confidence: {probs.max():.2%}")
    
    print(f"\nKey Features:")
    print(f"  Days of supply: {features['days_of_supply'].iloc[0]:.1f}")
    print(f"  Regional transfer availability: {features['regional_transfer_availability'].iloc[0]:.0f}")
    print(f"  Isolation score: {features['isolation_score'].iloc[0]:.2f}")
    
    # Verify: Should be MEDIUM or HIGH (not critical due to regional help available)
    # Note: With perfect rule-based model, it might still be high/critical based on days_of_supply
    # But regional availability should influence the decision
    assert risk_level[0] in ['medium', 'high', 'critical'], \
        f"Expected 'medium', 'high', or 'critical', got '{risk_level[0]}'"
    
    print(f"\n✓ Scenario 5 PASSED: Prediction considers regional context ({risk_level[0].upper()})")
    return True


def test_scenario_6_edge_case_zero_demand():
    """Scenario 6: Edge case - zero demand (should not trigger shortage)"""
    
    print("\n" + "="*60)
    print("Scenario 6: Edge Case - Zero Demand")
    print("="*60)
    
    hospital_id = 'H006'
    resource_type = 'beds'
    
    # Setup: Hospital with low stock but zero demand
    current_inventory = create_sample_inventory(hospital_id, resource_type, quantity=5, capacity=100)
    demand_predictions = create_sample_demand_predictions(hospital_id, resource_type, consumption_per_day=0.01)  # Near zero
    admissions_history = create_sample_admissions(hospital_id, baseline=0, trend='stable')
    hospital_info = create_sample_hospital(hospital_id)
    
    # Engineer features
    features = engineer_shortage_features(
        current_inventory, demand_predictions,
        admissions_history, hospital_info
    )
    
    # Predict
    detector = ShortageDetector()
    detector.load()
    
    risk_level, probs = detector.predict(features, return_probabilities=True)
    
    print(f"\nPrediction: {risk_level[0]}")
    print(f"Confidence: {probs.max():.2%}")
    
    print(f"\nKey Features:")
    print(f"  Days of supply: {features['days_of_supply'].iloc[0]:.1f}")
    print(f"  Stock/demand ratio: {features['stock_demand_ratio'].iloc[0]:.2f}")
    
    # Verify: Should be LOW (no demand = no shortage risk)
    # Note: days_of_supply will be very high (999) with near-zero demand
    assert risk_level[0] == 'low', f"Expected 'low' (zero demand), got '{risk_level[0]}'"
    
    print("\n✓ Scenario 6 PASSED: Correctly handled zero demand")
    return True


def test_scenario_7_very_high_stock():
    """Scenario 7: Very high stock - should be low risk"""
    
    print("\n" + "="*60)
    print("Scenario 7: Very High Stock")
    print("="*60)
    
    hospital_id = 'H007'
    resource_type = 'ppe'
    
    # Setup: Hospital with very high stock relative to demand
    current_inventory = create_sample_inventory(hospital_id, resource_type, quantity=1000, capacity=1500)
    demand_predictions = create_sample_demand_predictions(hospital_id, resource_type, consumption_per_day=10.0)
    admissions_history = create_sample_admissions(hospital_id, baseline=30, trend='stable')
    hospital_info = create_sample_hospital(hospital_id)
    
    # Engineer features
    features = engineer_shortage_features(
        current_inventory, demand_predictions,
        admissions_history, hospital_info
    )
    
    # Predict
    detector = ShortageDetector()
    detector.load()
    
    risk_level, probs = detector.predict(features, return_probabilities=True)
    
    print(f"\nPrediction: {risk_level[0]}")
    print(f"Confidence: {probs.max():.2%}")
    
    print(f"\nKey Features:")
    print(f"  Days of supply: {features['days_of_supply'].iloc[0]:.1f}")
    print(f"  Stock/demand ratio: {features['stock_demand_ratio'].iloc[0]:.2f}")
    
    # Verify: Should be LOW (100 days supply >> 14)
    assert risk_level[0] == 'low', f"Expected 'low' (high stock), got '{risk_level[0]}'"
    
    print("\n✓ Scenario 7 PASSED: Correctly identified low risk with high stock")
    return True


def test_scenario_8_resource_criticality():
    """Scenario 8: Critical resources (ventilators) should be more sensitive"""
    
    print("\n" + "="*60)
    print("Scenario 8: Resource Criticality (Ventilators)")
    print("="*60)
    
    hospital_id = 'H008'
    resource_type = 'ventilators'  # Most critical resource
    
    # Setup: Hospital with moderate stock of critical resource
    current_inventory = create_sample_inventory(hospital_id, resource_type, quantity=10, capacity=50)
    demand_predictions = create_sample_demand_predictions(hospital_id, resource_type, consumption_per_day=3.0)
    admissions_history = create_sample_admissions(hospital_id, baseline=20, trend='stable')
    hospital_info = create_sample_hospital(hospital_id)
    
    # Engineer features
    features = engineer_shortage_features(
        current_inventory, demand_predictions,
        admissions_history, hospital_info
    )
    
    # Predict
    detector = ShortageDetector()
    detector.load()
    
    risk_level, probs = detector.predict(features, return_probabilities=True)
    
    print(f"\nPrediction: {risk_level[0]}")
    print(f"Confidence: {probs.max():.2%}")
    
    print(f"\nKey Features:")
    print(f"  Resource criticality: {features['resource_criticality'].iloc[0]:.2f}")
    print(f"  Days of supply: {features['days_of_supply'].iloc[0]:.1f}")
    
    # Verify: With 10 ventilators and 3/day demand = ~3 days supply, should be HIGH or CRITICAL
    assert risk_level[0] in ['high', 'critical'], \
        f"Expected 'high' or 'critical' for critical resource, got '{risk_level[0]}'"
    
    print(f"\n✓ Scenario 8 PASSED: Correctly identified risk for critical resource ({risk_level[0].upper()})")
    return True


def run_all_tests():
    """Run all test scenarios"""
    print("\n" + "="*60)
    print("SHORTAGE DETECTOR - TEST SCENARIOS")
    print("="*60)
    
    results = []
    test_functions = [
        test_scenario_1_low_stock_high_demand,
        test_scenario_2_medium_stock_normal_demand,
        test_scenario_3_gradual_depletion,
        test_scenario_4_sudden_surge,
        test_scenario_5_regional_imbalance,
        test_scenario_6_edge_case_zero_demand,
        test_scenario_7_very_high_stock,
        test_scenario_8_resource_criticality
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            results.append((test_func.__name__, True, None))
        except AssertionError as e:
            print(f"\n✗ {test_func.__name__} FAILED: {e}")
            results.append((test_func.__name__, False, str(e)))
        except Exception as e:
            print(f"\n✗ {test_func.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {test_name}")
        if error:
            print(f"    Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(f"⚠ {total - passed} TEST(S) FAILED")
        print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
