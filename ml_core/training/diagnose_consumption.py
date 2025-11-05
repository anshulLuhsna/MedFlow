"""
Diagnostic script to check if consumption predictions are reasonable
"""

import sys
import os
from pathlib import Path

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add paths
ml_core_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ml_core_dir))
sys.path.insert(0, str(ml_core_dir.parent))

from models.demand_forecaster import DemandForecaster
from utils.data_loader import DataLoader
from config import DEMAND_FORECAST_CONFIG
import numpy as np
from sklearn.model_selection import train_test_split

print("="*70)
print("CONSUMPTION FORECASTING DIAGNOSTIC")
print("="*70)

# Load data
print("\n1. Loading data...")
loader = DataLoader()
X, y, metadata = loader.prepare_training_data(
    'ventilators',
    sequence_length=DEMAND_FORECAST_CONFIG['sequence_length'],
    verbose=False
)

print(f"   Data shapes: X={X.shape}, y={y.shape}")
print(f"\n2. Target variable (y) statistics:")
print(f"   Min: {y.min():.2f}")
print(f"   Max: {y.max():.2f}")
print(f"   Mean: {y.mean():.2f}")
print(f"   Median: {np.median(y):.2f}")
print(f"   Std: {y.std():.2f}")
print(f"\n   Are values near zero? {y.mean() < 1.0}")
print(f"   Sample y values (first 5 samples, day 1): {y[:5, 0]}")

# Check input features
print(f"\n3. Input features (X) - last timestep of first sequence:")
feature_names = ['quantity', 'consumption', 'resupply', 'admissions',
                 'qty_ma7', 'qty_ma14', 'qty_trend', 'cons_trend',
                 'qty_chg', 'cons_chg', 'qty_per_adm', 'cons_rate',
                 'qty_mom', 'cons_mom', 'qty_pct', 'cons_pct', 'trend_dir']

for i, name in enumerate(feature_names):
    val = X[0, -1, i]
    print(f"   {i:2d}. {name:12s}: {val:8.2f}")

# Load model and predict
print(f"\n4. Loading trained model...")
forecaster = DemandForecaster('ventilators')
forecaster.load()

# Split data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_test_small = X_test[:10]  # Just 10 samples
y_test_small = y_test[:10]

print(f"\n5. Making predictions on {len(X_test_small)} samples...")
predictions = forecaster.predict(X_test_small, return_confidence=False)

print(f"\n6. Prediction analysis:")
print(f"   Predictions shape: {predictions.shape}")
print(f"   Predictions min: {predictions.min():.2f}")
print(f"   Predictions max: {predictions.max():.2f}")
print(f"   Predictions mean: {predictions.mean():.2f}")
print(f"   Predictions std: {predictions.std():.2f}")

# Compare actual vs predicted for first 3 samples
print(f"\n7. Sample comparisons (Day 1 forecast only):")
print(f"   {'Sample':<8} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'%Error':<10}")
print(f"   {'-'*60}")

for i in range(min(3, len(y_test_small))):
    actual = y_test_small[i, 0]
    pred = predictions[i, 0]
    error = abs(actual - pred)
    pct_error = (error / max(actual, 0.01)) * 100  # Avoid div by zero

    print(f"   #{i+1:<7} {actual:>9.2f} {pred:>10.2f} {error:>9.2f} {pct_error:>9.1f}%")

# Check if predictions are all zeros or near-constant
unique_preds = len(np.unique(predictions.round(2)))
print(f"\n8. Model diversity check:")
print(f"   Unique predictions (rounded to 2 decimals): {unique_preds}")
if unique_preds < 10:
    print(f"   ⚠️  WARNING: Model is producing very similar predictions!")
    print(f"   This suggests the model may not have learned properly.")

# Calculate proper metrics
mae = np.mean(np.abs(predictions - y_test_small))
rmse = np.sqrt(np.mean((predictions - y_test_small) ** 2))

# Better MAPE calculation
# Only calculate MAPE where actual consumption > 0.1 to avoid division issues
mask = y_test_small > 0.1
if np.sum(mask) > 0:
    mape_safe = np.mean(np.abs((y_test_small[mask] - predictions[mask]) / y_test_small[mask])) * 100
    print(f"\n9. Metrics (on samples with consumption > 0.1):")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE (safe): {mape_safe:.2f}%")
    print(f"   Samples used for MAPE: {np.sum(mask)} / {mask.size}")
else:
    print(f"\n9. Metrics:")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   ⚠️  Cannot calculate MAPE - all consumption values too small")

print(f"\n{'='*70}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*70}\n")
