"""
Calibrated PPE Forecaster

Automatically applies post-hoc calibration to achieve proper coverage.
Use this instead of DemandForecaster for production predictions.
"""

from ml_core.models.demand_forecaster import DemandForecaster


class CalibratedPPEForecaster:
    """
    PPE forecaster with automatic calibration adjustment

    Calibration factor: 1.143
    Target 80% CI coverage: 80%
    """

    SCALE_FACTOR = 1.143

    def __init__(self):
        self.forecaster = DemandForecaster('ppe')

    def load(self):
        """Load the underlying model"""
        self.forecaster.load()

    def predict(self, X, probabilistic=True, n_samples=200):
        """
        Make calibrated predictions

        Args:
            X: Input features
            probabilistic: If True, return calibrated intervals
            n_samples: Number of MC samples

        Returns:
            Predictions with calibrated intervals
        """
        # Get raw predictions
        pred = self.forecaster.predict(X, probabilistic=probabilistic, n_samples=n_samples)

        if not probabilistic:
            return pred

        # Apply calibration
        mean = pred['mean']

        pred['p10'] = mean + (pred['p10'] - mean) * self.SCALE_FACTOR
        pred['p25'] = mean + (pred['p25'] - mean) * self.SCALE_FACTOR
        pred['p75'] = mean + (pred['p75'] - mean) * self.SCALE_FACTOR
        pred['p90'] = mean + (pred['p90'] - mean) * self.SCALE_FACTOR
        pred['std'] = pred['std'] * self.SCALE_FACTOR

        return pred


# Example usage
if __name__ == "__main__":
    import numpy as np

    forecaster = CalibratedPPEForecaster()
    forecaster.load()

    # Make prediction
    X = np.random.randn(1, 30, 17)  # Mock data
    pred = forecaster.predict(X, probabilistic=True)

    print(f"Tomorrow's PPE forecast:")
    print(f"  Expected: {pred['mean'][0, 0]:.1f}")
    print(f"  80% CI: [{pred['p10'][0, 0]:.1f}, {pred['p90'][0, 0]:.1f}]")
    print(f"  Conservative (P90): {pred['p90'][0, 0]:.1f}")
