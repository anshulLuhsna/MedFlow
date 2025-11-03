"""
Test script for probabilistic demand forecasting
Evaluates MC Dropout predictions and calibration
"""

import sys
import os
from pathlib import Path

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add ml_core directory to Python path
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

parent_dir = ml_core_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from models.demand_forecaster import DemandForecaster
from utils.data_loader import DataLoader
from config import DEMAND_FORECAST_CONFIG
import numpy as np
from sklearn.model_selection import train_test_split


def calculate_calibration(y_true, percentile_preds, percentile_level):
    """
    Calculate calibration: % of actual values below the predicted percentile

    Args:
        y_true: Actual values
        percentile_preds: Predicted percentile
        percentile_level: Expected level (e.g., 0.10 for P10, 0.90 for P90)

    Returns:
        Actual coverage (should be close to percentile_level)
    """
    coverage = np.mean(y_true < percentile_preds)
    return coverage


def calculate_sharpness(p10, p90):
    """
    Calculate sharpness: Average width of 80% confidence interval

    Args:
        p10: 10th percentile predictions
        p90: 90th percentile predictions

    Returns:
        Average interval width
    """
    return np.mean(p90 - p10)


def test_probabilistic_forecast(resource_type: str):
    """
    Test probabilistic forecasting for a resource type

    Args:
        resource_type: Type of resource to test
    """
    print(f"\n{'='*70}")
    print(f"Testing Probabilistic Forecasting: {resource_type.upper()}")
    print(f"{'='*70}\n")

    # Initialize
    data_loader = DataLoader()
    forecaster = DemandForecaster(resource_type)

    # Load trained model
    print("Loading trained model...")
    forecaster.load()
    print("✓ Model loaded\n")

    # Load test data
    print("Loading test data...")
    X, y, metadata = data_loader.prepare_training_data(
        resource_type=resource_type,
        sequence_length=DEMAND_FORECAST_CONFIG['sequence_length'],
        verbose=False
    )

    # Use small test set for speed (MC Dropout is slow)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42  # Only 5% for testing
    )

    print(f"Test set: {len(X_test)} samples")
    print(f"Shape: X={X_test.shape}, y={y_test.shape}\n")

    # Run probabilistic prediction
    print("Running probabilistic forecast (this may take 1-2 minutes)...")
    prob_pred = forecaster.predict(
        X_test,
        probabilistic=True,
        n_samples=200  # Increased to 200 MC samples for better calibration
    )

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Extract predictions
    mean_pred = prob_pred['mean']
    median_pred = prob_pred['median']
    std_pred = prob_pred['std']
    p10_pred = prob_pred['p10']
    p25_pred = prob_pred['p25']
    p50_pred = prob_pred['p50']
    p75_pred = prob_pred['p75']
    p90_pred = prob_pred['p90']

    print(f"\n1. SUMMARY STATISTICS")
    print(f"   Mean prediction: {np.mean(mean_pred):.2f} ± {np.std(mean_pred):.2f}")
    print(f"   Median prediction: {np.mean(median_pred):.2f}")
    print(f"   Average uncertainty (std): {np.mean(std_pred):.2f}")
    print(f"   Average 80% CI width: {np.mean(p90_pred - p10_pred):.2f}")

    # Calibration analysis
    print(f"\n2. CALIBRATION (Predicted percentiles vs. actual coverage)")
    print(f"   {'Percentile':<15} {'Expected':<12} {'Actual':<12} {'Error':<10} Status")
    print(f"   {'-'*60}")

    percentiles = [
        ('P10', 0.10, p10_pred),
        ('P25', 0.25, p25_pred),
        ('P50 (Median)', 0.50, p50_pred),
        ('P75', 0.75, p75_pred),
        ('P90', 0.90, p90_pred)
    ]

    calibration_errors = []

    for name, expected, pred in percentiles:
        actual = calculate_calibration(y_test, pred, expected)
        error = abs(actual - expected)
        calibration_errors.append(error)

        # Status: Good if within ±5%, Warning if ±5-10%, Bad if >10%
        if error < 0.05:
            status = "✅ Good"
        elif error < 0.10:
            status = "⚠️  Warning"
        else:
            status = "❌ Bad"

        print(f"   {name:<15} {expected:>10.1%} {actual:>11.1%} {error:>9.1%} {status}")

    avg_calibration_error = np.mean(calibration_errors)
    print(f"\n   Average calibration error: {avg_calibration_error:.1%}")

    if avg_calibration_error < 0.05:
        print(f"   Overall calibration: ✅ Excellent")
    elif avg_calibration_error < 0.10:
        print(f"   Overall calibration: ⚠️  Acceptable")
    else:
        print(f"   Overall calibration: ❌ Poor")

    # Sharpness analysis
    print(f"\n3. SHARPNESS (Confidence interval width)")
    sharpness_80 = calculate_sharpness(p10_pred, p90_pred)
    sharpness_50 = calculate_sharpness(p25_pred, p75_pred)

    print(f"   80% CI width (P10-P90): {sharpness_80:.2f}")
    print(f"   50% CI width (P25-P75): {sharpness_50:.2f}")
    print(f"   Coefficient of variation: {(np.mean(std_pred) / np.mean(mean_pred)):.1%}")

    # Magnitude accuracy (compared to point predictions)
    print(f"\n4. MAGNITUDE ACCURACY")

    # MAE for different estimators
    mae_mean = np.mean(np.abs(y_test - mean_pred))
    mae_median = np.mean(np.abs(y_test - median_pred))
    mae_p50 = np.mean(np.abs(y_test - p50_pred))

    print(f"   MAE (mean):   {mae_mean:.2f}")
    print(f"   MAE (median): {mae_median:.2f}")
    print(f"   MAE (P50):    {mae_p50:.2f}")

    # RMSE
    rmse_mean = np.sqrt(np.mean((y_test - mean_pred) ** 2))
    print(f"   RMSE (mean):  {rmse_mean:.2f}")

    # Coverage of confidence intervals
    print(f"\n5. INTERVAL COVERAGE (% of actuals within predicted intervals)")
    coverage_80 = np.mean((y_test >= p10_pred) & (y_test <= p90_pred))
    coverage_50 = np.mean((y_test >= p25_pred) & (y_test <= p75_pred))

    print(f"   80% CI: {coverage_80:.1%} (expected: 80%)")
    print(f"   50% CI: {coverage_50:.1%} (expected: 50%)")

    # Sample predictions
    print(f"\n6. SAMPLE PREDICTIONS (First 3 test samples, Day 1 forecast)")
    print(f"   {'Sample':<8} {'Actual':<10} {'Mean':<10} {'P10':<10} {'P50':<10} {'P90':<10}")
    print(f"   {'-'*60}")

    for i in range(min(3, len(y_test))):
        actual = y_test[i, 0]  # First day of forecast
        mean = mean_pred[i, 0]
        p10 = p10_pred[i, 0]
        p50 = p50_pred[i, 0]
        p90 = p90_pred[i, 0]

        # Check if actual is within 80% CI
        in_ci = "✅" if p10 <= actual <= p90 else "❌"

        print(f"   #{i+1:<7} {actual:>9.1f} {mean:>9.1f} {p10:>9.1f} {p50:>9.1f} {p90:>9.1f} {in_ci}")

    # Summary verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if avg_calibration_error < 0.05 and abs(coverage_80 - 0.80) < 0.10:
        print("✅ Probabilistic forecasting is WELL-CALIBRATED and useful!")
        print("   → Confidence intervals are reliable")
        print("   → Can be used for risk-based planning")
    elif avg_calibration_error < 0.10 and abs(coverage_80 - 0.80) < 0.15:
        print("⚠️  Probabilistic forecasting is ACCEPTABLE but needs improvement")
        print("   → Calibration is reasonable but not perfect")
        print("   → Can be used with caution")
    else:
        print("❌ Probabilistic forecasting is POORLY CALIBRATED")
        print("   → Confidence intervals are not reliable")
        print("   → Needs further tuning (more dropout, more samples, etc.)")

    print(f"{'='*70}\n")

    return {
        'resource_type': resource_type,
        'calibration_error': avg_calibration_error,
        'coverage_80': coverage_80,
        'sharpness_80': sharpness_80,
        'mae_mean': mae_mean,
        'rmse_mean': rmse_mean
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test probabilistic forecasting')
    parser.add_argument(
        '--resource',
        type=str,
        default='ventilators',
        help='Resource type to test (default: ventilators)'
    )

    args = parser.parse_args()

    result = test_probabilistic_forecast(args.resource)

    print(f"\nTest complete! Results saved in memory.")
    print(f"Calibration error: {result['calibration_error']:.1%}")
    print(f"80% CI coverage: {result['coverage_80']:.1%}")
