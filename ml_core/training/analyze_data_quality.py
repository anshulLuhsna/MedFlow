"""
Comprehensive data quality analysis for ML training
Checks if synthetic data is suitable for training demand forecasting models
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

from utils.data_loader import DataLoader
from config import DEMAND_FORECAST_CONFIG, RESOURCE_TYPES
import numpy as np

print("="*80)
print("DATA QUALITY ANALYSIS FOR ML TRAINING")
print("="*80)

loader = DataLoader()

# Analyze all resources
for resource_type in RESOURCE_TYPES:
    print(f"\n{'='*80}")
    print(f"RESOURCE: {resource_type.upper()}")
    print(f"{'='*80}")

    try:
        # Load data
        X, y, metadata = loader.prepare_training_data(
            resource_type=resource_type,
            sequence_length=DEMAND_FORECAST_CONFIG['sequence_length'],
            verbose=False
        )

        print(f"\n1. DATA VOLUME")
        print(f"   Samples: {len(y):,}")
        print(f"   Shape: X={X.shape}, y={y.shape}")

        print(f"\n2. TARGET (y) - CONSUMPTION STATISTICS")
        print(f"   Mean:           {y.mean():.2f}")
        print(f"   Median:         {np.median(y):.2f}")
        print(f"   Std Dev:        {y.std():.2f}")
        print(f"   Min:            {y.min():.2f}")
        print(f"   Max:            {y.max():.2f}")
        print(f"   25th percentile: {np.percentile(y, 25):.2f}")
        print(f"   75th percentile: {np.percentile(y, 75):.2f}")

        print(f"\n3. SPARSITY ANALYSIS")
        zero_count = (y == 0).sum()
        total_count = y.size
        zero_pct = (zero_count / total_count) * 100

        print(f"   Zero values:    {zero_count:,} / {total_count:,} ({zero_pct:.1f}%)")
        print(f"   Non-zero:       {total_count - zero_count:,} ({100-zero_pct:.1f}%)")

        # Check different thresholds
        above_0_1 = (y > 0.1).sum()
        above_1 = (y > 1.0).sum()
        above_5 = (y > 5.0).sum()
        above_10 = (y > 10.0).sum()

        print(f"   Above 0.1:      {above_0_1:,} ({above_0_1/total_count*100:.1f}%)")
        print(f"   Above 1.0:      {above_1:,} ({above_1/total_count*100:.1f}%)")
        print(f"   Above 5.0:      {above_5:,} ({above_5/total_count*100:.1f}%)")
        print(f"   Above 10.0:     {above_10:,} ({above_10/total_count*100:.1f}%)")

        print(f"\n4. VARIATION ANALYSIS")
        # Check day-to-day variation
        y_diff = np.diff(y, axis=1)
        print(f"   Day-to-day changes:")
        print(f"     Mean abs change:  {np.abs(y_diff).mean():.2f}")
        print(f"     Std of changes:   {y_diff.std():.2f}")

        # Direction changes
        increases = (y_diff > 0).sum()
        decreases = (y_diff < 0).sum()
        no_change = (y_diff == 0).sum()

        print(f"   Direction distribution:")
        print(f"     Increases:  {increases:,} ({increases/y_diff.size*100:.1f}%)")
        print(f"     Decreases:  {decreases:,} ({decreases/y_diff.size*100:.1f}%)")
        print(f"     No change:  {no_change:,} ({no_change/y_diff.size*100:.1f}%)")

        print(f"\n5. INPUT FEATURES - CORRELATION WITH TARGET")
        feature_names = ['quantity', 'consumption', 'resupply', 'admissions',
                        'qty_ma7', 'qty_ma14', 'qty_trend', 'cons_trend',
                        'qty_chg', 'cons_chg', 'qty_per_adm', 'cons_rate',
                        'qty_mom', 'cons_mom', 'qty_pct', 'cons_pct', 'trend_dir']

        # Average over sequence length for correlation
        X_avg = X.mean(axis=1)  # Average across time dimension
        y_avg = y.mean(axis=1)  # Average across forecast horizon

        correlations = []
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X_avg[:, i], y_avg)[0, 1]
            if not np.isnan(corr):
                correlations.append((name, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"   Top 5 correlated features:")
        for name, corr in correlations[:5]:
            print(f"     {name:12s}: {corr:+.3f}")

        print(f"\n6. ML TRAINING SUITABILITY")

        # Criteria for good ML training data
        criteria = []

        # 1. Sufficient non-zero values (>50%)
        if zero_pct < 50:
            criteria.append(("✅", f"Non-zero values: {100-zero_pct:.1f}% (good)"))
        else:
            criteria.append(("❌", f"Too sparse: {zero_pct:.1f}% zeros (bad)"))

        # 2. Reasonable mean (>1.0 for most resources)
        if y.mean() >= 1.0:
            criteria.append(("✅", f"Mean consumption: {y.mean():.1f} (good)"))
        elif y.mean() >= 0.5:
            criteria.append(("⚠️ ", f"Mean consumption: {y.mean():.2f} (marginal)"))
        else:
            criteria.append(("❌", f"Mean too low: {y.mean():.2f} (bad)"))

        # 3. Sufficient variation (CV > 0.3)
        cv = y.std() / max(y.mean(), 0.01)  # Coefficient of variation
        if cv > 0.3:
            criteria.append(("✅", f"Variation (CV): {cv:.2f} (good)"))
        else:
            criteria.append(("⚠️ ", f"Low variation (CV): {cv:.2f} (poor)"))

        # 4. Direction changes (both up and down)
        if increases > 0.2 * y_diff.size and decreases > 0.2 * y_diff.size:
            criteria.append(("✅", "Balanced directional changes (good)"))
        else:
            criteria.append(("⚠️ ", "Imbalanced directional changes"))

        # 5. Strong correlation with admissions
        admission_corr = next((corr for name, corr in correlations if 'admission' in name), 0)
        if abs(admission_corr) > 0.5:
            criteria.append(("✅", f"Admission correlation: {admission_corr:.2f} (good)"))
        elif abs(admission_corr) > 0.3:
            criteria.append(("⚠️ ", f"Admission correlation: {admission_corr:.2f} (marginal)"))
        else:
            criteria.append(("❌", f"Admission correlation: {admission_corr:.2f} (bad)"))

        print(f"   Assessment:")
        for symbol, message in criteria:
            print(f"     {symbol} {message}")

        # Overall verdict
        good_count = sum(1 for s, _ in criteria if s == "✅")
        warning_count = sum(1 for s, _ in criteria if s == "⚠️ ")
        bad_count = sum(1 for s, _ in criteria if s == "❌")

        print(f"\n   VERDICT:")
        if good_count >= 4:
            print(f"     ✅ EXCELLENT - Ready for ML training")
        elif good_count >= 3:
            print(f"     ✅ GOOD - Suitable for ML training")
        elif good_count + warning_count >= 4:
            print(f"     ⚠️  MARGINAL - May need data improvements")
        else:
            print(f"     ❌ POOR - Data generation needs fixing")

    except Exception as e:
        print(f"\n   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")

print("RECOMMENDATIONS:")
print("  - Resources with 'EXCELLENT' or 'GOOD' verdict: Train immediately")
print("  - Resources with 'MARGINAL' verdict: May work but expect lower accuracy")
print("  - Resources with 'POOR' verdict: Fix data generation before training")
print()
