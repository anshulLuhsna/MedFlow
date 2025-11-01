"""
Post-hoc calibration adjustment for probabilistic forecasts

When MC Dropout produces poorly calibrated intervals, we can apply
temperature scaling to widen them without retraining.
"""

import numpy as np


class ProbabilisticCalibrator:
    """
    Adjusts probabilistic forecasts to improve calibration

    Uses empirical calibration factor to widen/narrow intervals
    """

    def __init__(self):
        self.calibration_factor = None
        self.is_fitted = False

    def fit(self, y_true, predictions, target_percentile=0.9):
        """
        Learn calibration factor from validation data

        Args:
            y_true: Actual values (n_samples, forecast_horizon)
            predictions: MC Dropout samples (n_mc_samples, n_samples, forecast_horizon)
            target_percentile: Percentile to calibrate (default 90th)

        Returns:
            self
        """
        # Calculate empirical coverage at target percentile
        p_pred = np.percentile(predictions, target_percentile * 100, axis=0)
        empirical_coverage = np.mean(y_true < p_pred)

        # If coverage is too low (intervals too narrow), we need to widen
        # If coverage is too high (intervals too wide), we need to narrow

        # Calculate adjustment factor
        # E.g., if P90 only covers 82% instead of 90%, we need to widen
        if empirical_coverage < target_percentile:
            # Intervals too narrow - increase spread
            self.calibration_factor = target_percentile / max(empirical_coverage, 0.01)
        else:
            # Intervals too wide - decrease spread
            self.calibration_factor = target_percentile / empirical_coverage

        self.is_fitted = True

        print(f"Calibration factor: {self.calibration_factor:.3f}")
        print(f"  Empirical {target_percentile:.0%} coverage: {empirical_coverage:.1%}")
        print(f"  Target coverage: {target_percentile:.0%}")

        return self

    def adjust(self, predictions):
        """
        Apply calibration adjustment to new predictions

        Args:
            predictions: Dict with 'mean', 'std', 'p10', 'p50', 'p90', etc.

        Returns:
            Adjusted predictions dict
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        # Don't adjust mean/median
        adjusted = predictions.copy()

        # Adjust percentiles by scaling around the mean
        for key in ['p10', 'p25', 'p75', 'p90']:
            if key in predictions:
                mean = predictions['mean']
                deviation = predictions[key] - mean
                adjusted[key] = mean + (deviation * self.calibration_factor)

        # Adjust std
        if 'std' in predictions:
            adjusted['std'] = predictions['std'] * self.calibration_factor

        return adjusted


def simple_calibration_adjustment(predictions, scale_factor=1.3):
    """
    Quick calibration fix without fitting

    Args:
        predictions: Dict with probabilistic forecasts
        scale_factor: How much to widen intervals (1.3 = 30% wider)

    Returns:
        Adjusted predictions

    Example:
        # If 80% CI only covers 60%, use scale_factor ~1.3
        adjusted = simple_calibration_adjustment(pred, scale_factor=1.3)
    """
    adjusted = predictions.copy()
    mean = predictions['mean']

    # Widen all percentiles proportionally
    for key in ['p10', 'p25', 'p75', 'p90']:
        if key in predictions:
            deviation = predictions[key] - mean
            adjusted[key] = mean + (deviation * scale_factor)

    if 'std' in predictions:
        adjusted['std'] = predictions['std'] * scale_factor

    return adjusted


def calculate_optimal_scale_factor(y_true, predictions, target_coverage=0.8):
    """
    Calculate optimal scale factor for calibration

    Args:
        y_true: Actual values
        predictions: Dict with p10, p90
        target_coverage: Desired coverage (default 80%)

    Returns:
        Optimal scale factor
    """
    # Current 80% CI coverage
    p10 = predictions['p10']
    p90 = predictions['p90']
    current_coverage = np.mean((y_true >= p10) & (y_true <= p90))

    # If current coverage is too low, we need to widen
    if current_coverage < target_coverage:
        # Estimate scale factor needed
        # This is approximate - assumes linear relationship
        scale_factor = np.sqrt(target_coverage / max(current_coverage, 0.01))
    else:
        scale_factor = 1.0

    print(f"Current 80% CI coverage: {current_coverage:.1%}")
    print(f"Target coverage: {target_coverage:.0%}")
    print(f"Recommended scale factor: {scale_factor:.2f}")

    return scale_factor


# Example usage
if __name__ == "__main__":
    # Simulate predictions
    np.random.seed(42)
    n_samples = 100

    # True values
    y_true = np.random.randn(n_samples, 14) * 5 + 10

    # Simulated MC Dropout predictions (too narrow)
    mc_samples = 100
    predictions_mc = y_true[np.newaxis, :, :] + np.random.randn(mc_samples, n_samples, 14) * 2

    # Calculate percentiles
    predictions = {
        'mean': np.mean(predictions_mc, axis=0),
        'std': np.std(predictions_mc, axis=0),
        'p10': np.percentile(predictions_mc, 10, axis=0),
        'p50': np.percentile(predictions_mc, 50, axis=0),
        'p90': np.percentile(predictions_mc, 90, axis=0)
    }

    # Check coverage before
    coverage_before = np.mean((y_true >= predictions['p10']) & (y_true <= predictions['p90']))
    print(f"Coverage before adjustment: {coverage_before:.1%}")

    # Calculate optimal scale factor
    scale_factor = calculate_optimal_scale_factor(y_true, predictions)

    # Apply adjustment
    adjusted = simple_calibration_adjustment(predictions, scale_factor)

    # Check coverage after
    coverage_after = np.mean((y_true >= adjusted['p10']) & (y_true <= adjusted['p90']))
    print(f"Coverage after adjustment: {coverage_after:.1%}")
