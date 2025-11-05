"""
Demand Forecasting Model - LSTM Neural Network
Predicts future resource CONSUMPTION based on historical patterns

Target Variable: CONSUMPTION (actual demand) - NOT quantity (stock levels)
- Consumption is predictable from admissions and historical patterns
- Quantity includes random resupply decisions and is not directly predictable
- Model forecasts 14-day consumption to inform procurement planning
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import os

# Configure TensorFlow GPU settings before import
# Prevents "libdevice not found" errors by using CPU if CUDA not properly configured
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU by default

import tensorflow as tf

# Disable GPU if environment variable is set
if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass  # GPU already disabled or not available

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path

# Handle both relative and absolute imports
try:
    from ..config import DEMAND_FORECAST_CONFIG, MODEL_PATHS, RESOURCE_TYPES
except ImportError:
    from config import DEMAND_FORECAST_CONFIG, MODEL_PATHS, RESOURCE_TYPES


def create_directional_loss(magnitude_weight=0.7, direction_weight=0.3):
    """
    Create a combined loss function that optimizes both magnitude accuracy and directional accuracy.

    Problem: Standard losses (MSE, MAE, Huber) only care about how close predictions are to actual values.
    They don't care if the model predicts demand is going UP when it's actually going DOWN.

    Solution: Add a directional component that rewards the model for predicting the correct trend direction.

    Args:
        magnitude_weight: Weight for magnitude accuracy (default 0.7 = 70%)
        direction_weight: Weight for directional accuracy (default 0.3 = 30%)

    Returns:
        Combined loss function
    """
    def directional_loss(y_true, y_pred):
        """
        Combined loss: magnitude (Huber) + directional accuracy

        The loss has two components:
        1. Magnitude: How far off are the predictions? (Huber loss)
        2. Direction: Are we predicting the right trend direction? (Custom directional penalty)
        """
        # Component 1: Magnitude accuracy using Huber loss (robust to outliers)
        huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)

        # Component 2: Directional accuracy
        # Calculate day-to-day changes (trends) across the forecast horizon
        # Shape: (batch_size, forecast_horizon-1)
        y_true_diff = y_true[:, 1:] - y_true[:, :-1]  # Actual trend direction
        y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]  # Predicted trend direction

        # Check if signs match (both increasing, both decreasing, or both flat)
        # When signs match: direction_match = 1.0 (good)
        # When signs differ: direction_match = -1.0 (bad)
        # When one is zero: direction_match = 0.0 (neutral)
        direction_match = tf.sign(y_true_diff) * tf.sign(y_pred_diff)

        # Convert to penalty:
        # - Perfect match (1.0) → penalty = 0.0
        # - Wrong direction (-1.0) → penalty = 2.0
        # - Neutral (0.0) → penalty = 1.0
        directional_penalty = tf.reduce_mean(tf.maximum(0.0, 1.0 - direction_match))

        # Combine: 70% magnitude + 30% direction (configurable)
        combined_loss = magnitude_weight * huber + direction_weight * directional_penalty

        return combined_loss

    return directional_loss


def directional_accuracy_metric(y_true, y_pred):
    """
    Metric to track directional accuracy during training.

    This answers: "What percentage of the time does the model correctly predict
    whether demand will increase or decrease compared to the previous day?"

    Returns:
        Percentage of correct directional predictions (0.0 to 1.0)
    """
    # Calculate day-to-day changes
    y_true_diff = y_true[:, 1:] - y_true[:, :-1]
    y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]

    # Count when signs match (both positive, both negative, or both zero)
    correct_direction = tf.cast(
        tf.equal(tf.sign(y_true_diff), tf.sign(y_pred_diff)),
        tf.float32
    )

    # Return mean accuracy across all predictions
    return tf.reduce_mean(correct_direction)


class DemandForecaster:
    """
    LSTM-based demand forecasting model for medical resource consumption

    Predicts: CONSUMPTION (actual demand) - NOT quantity (stock levels)

    Why consumption?
    - Consumption is driven by patient admissions (predictable signal)
    - Quantity includes random resupply/procurement decisions (unpredictable)
    - For planning, we need to know expected USAGE, not stock fluctuations

    Input: 30 days of historical data (17 features including admissions, trends, etc.)
    Output: 14-day consumption forecast with uncertainty quantification
    """

    def __init__(self, resource_type: str):
        if resource_type not in RESOURCE_TYPES:
            raise ValueError(f"Invalid resource type: {resource_type}")
        
        self.resource_type = resource_type
        self.config = DEMAND_FORECAST_CONFIG
        self.model = None
        self.scaler = StandardScaler()  # For input features (X)
        self.y_scaler = StandardScaler()  # For targets (y) - NEW
        self.model_path = MODEL_PATHS["demand_forecaster"] / resource_type
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: (sequence_length, n_features)
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First LSTM layer
            layers.LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                dropout=self.config['dropout']
            ),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(
                self.config['lstm_units'] // 2,
                return_sequences=False,
                dropout=self.config['dropout']
            ),
            layers.BatchNormalization(),
            
            # Dense layers with dropout for MC Dropout uncertainty
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.config['dropout']),  # Dropout 1
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.config['dropout']),  # Dropout 2 - ADDED for better uncertainty

            # Output layer (predict next forecast_horizon days)
            layers.Dense(self.config['forecast_horizon'])
        ])
        
        # Add gradient clipping to prevent training instability/explosions
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=1.0  # Clip gradients to prevent explosions (like PPE epoch 15 issue)
        )

        # Use combined directional loss for better trend prediction
        # This optimizes both magnitude accuracy AND directional accuracy
        # 70% weight on magnitude (how close?), 30% weight on direction (up or down?)
        model.compile(
            optimizer=optimizer,
            loss=create_directional_loss(magnitude_weight=0.7, direction_weight=0.3),
            metrics=['mae', 'mape', directional_accuracy_metric]  # Track directional accuracy
        )
        
        return model
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize and prepare data
        
        Args:
            X: Input sequences (samples, sequence_length, features)
            y: Target values (samples, forecast_horizon)
        
        Returns:
            Normalized X and y
        """
        # Reshape X for scaling
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        # Fit and transform X
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        
        # Scale y targets (NEW)
        # StandardScaler expects (n_samples, n_features)
        # y shape is (n_samples, forecast_horizon), which is already correct
        original_y_shape = y.shape
        
        # Ensure y is a proper numpy array with correct dtype
        y = np.asarray(y, dtype=np.float64)
        if not y.flags['C_CONTIGUOUS']:
            y = np.ascontiguousarray(y)
        
        if len(y.shape) == 2:
            # Already in correct format (n_samples, forecast_horizon)
            y_reshaped = y  # No reshape needed
        elif len(y.shape) == 1:
            # 1D array, reshape to (n_samples, 1)
            y_reshaped = y.reshape(-1, 1)
        else:
            # Flatten to 2D if needed
            y_reshaped = y.reshape(-1, y.shape[-1]) if len(y.shape) > 2 else y.reshape(-1, 1)
        
        # Ensure C-contiguous for scikit-learn
        y_reshaped = np.ascontiguousarray(y_reshaped, dtype=np.float64)
        
        # Fit and transform
        y_scaled = self.y_scaler.fit_transform(y_reshaped)
        
        # Reshape back to original shape
        y_scaled = y_scaled.reshape(original_y_shape)
        
        return X_scaled, y_scaled
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the demand forecasting model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            verbose: Verbosity level
        
        Returns:
            Training history dict
        """
        # Prepare data (this scales both X and y)
        X_train_scaled, y_train_scaled = self.prepare_sequences(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            # Scale X_val using already-fitted scaler
            X_val_scaled = self.scaler.transform(
                X_val.reshape(-1, X_val.shape[2])
            ).reshape(X_val.shape)
            
            # Scale y_val using already-fitted y_scaler
            original_y_val_shape = y_val.shape
            
            # Ensure y_val is proper numpy array
            y_val = np.asarray(y_val, dtype=np.float64)
            if not y_val.flags['C_CONTIGUOUS']:
                y_val = np.ascontiguousarray(y_val)
            
            if len(y_val.shape) == 2:
                # Already in correct format
                y_val_reshaped = y_val
            elif len(y_val.shape) == 1:
                y_val_reshaped = y_val.reshape(-1, 1)
            else:
                y_val_reshaped = y_val.reshape(-1, y_val.shape[-1])
            
            # Ensure C-contiguous for scikit-learn
            y_val_reshaped = np.ascontiguousarray(y_val_reshaped, dtype=np.float64)
            
            y_val_scaled = self.y_scaler.transform(y_val_reshaped)
            
            # Reshape back to original shape
            y_val_scaled = y_val_scaled.reshape(original_y_val_shape)
            
            validation_data = (X_val_scaled, y_val_scaled)
        else:
            validation_data = None
        
        # Build model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        self.model = self.build_model(input_shape)
        
        if verbose:
            print(f"\nTraining Demand Forecaster for {self.resource_type}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {self.config['forecast_horizon']} days")
            self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.config['early_stopping_patience'],
                min_delta=self.config.get('early_stopping_min_delta', 0.001),
                restore_best_weights=True,
                verbose=1  # Show when early stopping triggers
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=4,  # Reduced from 5 for faster adaptation
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path / 'best_model.h5'),
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train_scaled,
            y_train_scaled,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Save model and scaler
        self.save()
        
        return history.history
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False,
        probabilistic: bool = False,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Predict future demand with optional probabilistic forecasting

        Args:
            X: Input sequences (samples, sequence_length, features)
            return_confidence: Whether to return prediction intervals (simple approach)
            probabilistic: Whether to use Monte Carlo Dropout for uncertainty (advanced)
            n_samples: Number of MC samples for probabilistic predictions (default: 100)

        Returns:
            If probabilistic=True: Dict with mean, std, percentiles
            If return_confidence=True: (predictions, lower_bound, upper_bound)
            Otherwise: Predictions (samples, forecast_horizon)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Scale input
        n_samples_batch, seq_len, n_features = X.shape
        X_scaled = self.scaler.transform(
            X.reshape(-1, n_features)
        ).reshape(n_samples_batch, seq_len, n_features)

        # Probabilistic prediction with Monte Carlo Dropout
        if probabilistic:
            return self._predict_probabilistic(X_scaled, n_samples)

        # Standard prediction (outputs are in scaled space)
        predictions_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Ensure predictions_scaled is a proper numpy array
        # Convert to numpy, ensure C-contiguous, and correct dtype
        if hasattr(predictions_scaled, 'numpy'):
            predictions_scaled = predictions_scaled.numpy()
        predictions_scaled = np.asarray(predictions_scaled, dtype=np.float64)
        if not predictions_scaled.flags['C_CONTIGUOUS']:
            predictions_scaled = np.ascontiguousarray(predictions_scaled)
        
        # Validate y_scaler has been fitted
        if not hasattr(self.y_scaler, 'mean_') or self.y_scaler.mean_ is None:
            raise ValueError("y_scaler not fitted. Call prepare_sequences() or train() first.")
        
        # Inverse transform predictions back to original scale
        # StandardScaler expects same shape as during fit: (n_samples, forecast_horizon)
        original_shape = predictions_scaled.shape
        
        if len(predictions_scaled.shape) == 2:
            # Already in correct format (n_samples, forecast_horizon)
            predictions_reshaped = predictions_scaled
        elif len(predictions_scaled.shape) == 1:
            # 1D array, reshape to (n_samples, 1)
            predictions_reshaped = predictions_scaled.reshape(-1, 1)
        else:
            # Flatten to 2D
            predictions_reshaped = predictions_scaled.reshape(-1, predictions_scaled.shape[-1])
        
        # Ensure proper numpy array for scikit-learn (C-contiguous, float64)
        predictions_reshaped = np.asarray(predictions_reshaped, dtype=np.float64, order='C')
        
        # Validate shape matches scaler expectations
        expected_features = len(self.y_scaler.mean_) if hasattr(self.y_scaler, 'mean_') else None
        if expected_features and predictions_reshaped.shape[1] != expected_features:
            raise ValueError(
                f"Shape mismatch: predictions have {predictions_reshaped.shape[1]} features, "
                f"but y_scaler expects {expected_features} features"
            )
        
        # Use manual inverse transform directly (sklearn has ArrayType issues with certain array types)
        # StandardScaler formula: X_original = X_scaled * scale_ + mean_
        if hasattr(self.y_scaler, 'mean_') and hasattr(self.y_scaler, 'scale_'):
            scale = np.asarray(self.y_scaler.scale_, dtype=np.float64)
            mean = np.asarray(self.y_scaler.mean_, dtype=np.float64)
            # Broadcasting: (n_samples, n_features) * (n_features,) + (n_features,)
            predictions = predictions_reshaped * scale + mean
        else:
            raise ValueError(
                f"y_scaler not properly fitted. Missing mean_ or scale_ attributes. "
                f"Call prepare_sequences() or train() first."
            )
        
        # Reshape back to original shape
        predictions = predictions.reshape(original_shape)
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        if return_confidence:
            # Simple confidence intervals (can be improved with MC Dropout)
            # Calculate intervals in original scale
            std = np.std(predictions, axis=0, keepdims=True)
            lower = predictions - 1.96 * std
            upper = predictions + 1.96 * std
            lower = np.maximum(lower, 0)  # Ensure non-negative
            return predictions, lower, upper
        
        return predictions

    def _predict_probabilistic(
        self,
        X_scaled: np.ndarray,
        n_samples: int = 100
    ) -> dict:
        """
        Generate probabilistic predictions using Monte Carlo Dropout

        Monte Carlo Dropout: Run inference multiple times with dropout enabled
        to sample from the implicit posterior distribution.

        Args:
            X_scaled: Scaled input sequences
            n_samples: Number of forward passes (default: 100)

        Returns:
            Dictionary with probabilistic predictions:
            {
                'mean': Mean prediction across samples,
                'median': Median (P50),
                'std': Standard deviation (uncertainty),
                'p10': 10th percentile,
                'p25': 25th percentile,
                'p50': 50th percentile (median),
                'p75': 75th percentile,
                'p90': 90th percentile,
                'samples': All prediction samples (for advanced use)
            }
        """
        print(f"Running probabilistic forecast with {n_samples} MC samples...")

        # Collect predictions from multiple forward passes with dropout enabled
        mc_predictions = []

        for i in range(n_samples):
            if i % 20 == 0 and i > 0:
                print(f"  Progress: {i}/{n_samples} samples...")

            # Key: training=True keeps dropout active during inference!
            # This creates model uncertainty by randomly dropping neurons
            pred_scaled = self.model(X_scaled, training=True)

            # Convert to numpy
            if hasattr(pred_scaled, 'numpy'):
                pred_scaled = pred_scaled.numpy()
            pred_scaled = np.asarray(pred_scaled, dtype=np.float64)

            # Inverse transform to original scale
            pred_original = self._inverse_transform_predictions(pred_scaled)

            mc_predictions.append(pred_original)

        # Stack all samples: (n_samples, batch_size, forecast_horizon)
        mc_predictions = np.array(mc_predictions)

        print(f"  ✓ Generated {n_samples} samples")

        # Calculate statistics across samples (axis=0)
        mean_pred = np.mean(mc_predictions, axis=0)
        median_pred = np.median(mc_predictions, axis=0)
        std_pred = np.std(mc_predictions, axis=0)

        # Calculate percentiles
        p10_pred = np.percentile(mc_predictions, 10, axis=0)
        p25_pred = np.percentile(mc_predictions, 25, axis=0)
        p50_pred = np.percentile(mc_predictions, 50, axis=0)  # Same as median
        p75_pred = np.percentile(mc_predictions, 75, axis=0)
        p90_pred = np.percentile(mc_predictions, 90, axis=0)

        # Ensure monotonicity: P10 ≤ P25 ≤ P50 ≤ P75 ≤ P90
        # (MC Dropout can sometimes violate this due to sampling noise)
        for i in range(p10_pred.shape[0]):  # For each batch sample
            for j in range(p10_pred.shape[1]):  # For each forecast day
                percentiles = np.array([
                    p10_pred[i, j],
                    p25_pred[i, j],
                    p50_pred[i, j],
                    p75_pred[i, j],
                    p90_pred[i, j]
                ])
                # Sort to ensure monotonicity
                percentiles_sorted = np.sort(percentiles)
                p10_pred[i, j] = percentiles_sorted[0]
                p25_pred[i, j] = percentiles_sorted[1]
                p50_pred[i, j] = percentiles_sorted[2]
                p75_pred[i, j] = percentiles_sorted[3]
                p90_pred[i, j] = percentiles_sorted[4]

        # Ensure all predictions are non-negative (demand can't be negative)
        mean_pred = np.maximum(mean_pred, 0)
        median_pred = np.maximum(median_pred, 0)
        p10_pred = np.maximum(p10_pred, 0)
        p25_pred = np.maximum(p25_pred, 0)
        p50_pred = np.maximum(p50_pred, 0)
        p75_pred = np.maximum(p75_pred, 0)
        p90_pred = np.maximum(p90_pred, 0)

        return {
            'mean': mean_pred,
            'median': median_pred,
            'std': std_pred,
            'p10': p10_pred,
            'p25': p25_pred,
            'p50': p50_pred,
            'p75': p75_pred,
            'p90': p90_pred,
            'samples': mc_predictions  # All samples for advanced analysis
        }

    def _inverse_transform_predictions(self, predictions_scaled: np.ndarray) -> np.ndarray:
        """
        Helper to inverse transform scaled predictions back to original scale

        Args:
            predictions_scaled: Predictions in scaled space

        Returns:
            Predictions in original scale
        """
        # Ensure proper numpy array
        predictions_scaled = np.asarray(predictions_scaled, dtype=np.float64)
        if not predictions_scaled.flags['C_CONTIGUOUS']:
            predictions_scaled = np.ascontiguousarray(predictions_scaled)

        # Validate y_scaler
        if not hasattr(self.y_scaler, 'mean_') or self.y_scaler.mean_ is None:
            raise ValueError("y_scaler not fitted. Call prepare_sequences() or train() first.")

        # Manual inverse transform: X_original = X_scaled * scale_ + mean_
        original_shape = predictions_scaled.shape

        if len(predictions_scaled.shape) == 2:
            predictions_reshaped = predictions_scaled
        elif len(predictions_scaled.shape) == 1:
            predictions_reshaped = predictions_scaled.reshape(-1, 1)
        else:
            predictions_reshaped = predictions_scaled.reshape(-1, predictions_scaled.shape[-1])

        predictions_reshaped = np.asarray(predictions_reshaped, dtype=np.float64, order='C')

        # Inverse transform
        scale = np.asarray(self.y_scaler.scale_, dtype=np.float64)
        mean = np.asarray(self.y_scaler.mean_, dtype=np.float64)
        predictions = predictions_reshaped * scale + mean

        # Reshape back
        predictions = predictions.reshape(original_shape)

        return predictions

    def predict_for_hospital(
        self,
        hospital_id: str,
        historical_data: pd.DataFrame,
        days_ahead: int = 14
    ) -> Dict:
        """
        Predict demand for a specific hospital
        
        Args:
            hospital_id: Hospital UUID
            historical_data: DataFrame with columns [date, quantity, consumption, resupply, total_admissions]
            days_ahead: Number of days to forecast
        
        Returns:
            Dict with predictions and metadata
        """
        # Ensure date is datetime for sorting
        if 'date' not in historical_data.columns:
            raise ValueError("historical_data must contain 'date' column")
        
        # Ensure resupply column exists (fill with 0 if missing)
        if 'resupply' not in historical_data.columns:
            historical_data['resupply'] = 0
        
        # Ensure total_admissions column exists (fill with 0 if missing)
        if 'total_admissions' not in historical_data.columns:
            historical_data['total_admissions'] = 0
        
        # Sort by date and get last sequence_length days
        data = historical_data.sort_values('date').copy()
        data = data.tail(self.config['sequence_length'])
        
        if len(data) < self.config['sequence_length']:
            raise ValueError(f"Need at least {self.config['sequence_length']} days of history")
        
        # Engineer all 17 features (same as training)
        from ..utils.feature_engineering import calculate_trend
        
        # Calculate rolling means for quantity (7-day and 14-day)
        data['quantity_ma_7d'] = data['quantity'].rolling(window=7, min_periods=1).mean()
        data['quantity_ma_14d'] = data['quantity'].rolling(window=14, min_periods=1).mean()
        
        # Calculate trend (slope over last 14 days)
        data['quantity_trend'] = calculate_trend(data['quantity'], window=14)
        data['consumption_trend'] = calculate_trend(data['consumption'], window=14)
        
        # Add rate of change features
        data['quantity_change'] = data['quantity'].diff().fillna(0)
        data['consumption_change'] = data['consumption'].diff().fillna(0)
        
        # Add relative features (normalized by admissions)
        data['quantity_per_admission'] = data['quantity'] / (data['total_admissions'] + 1)
        data['consumption_rate'] = data['consumption'] / (data['total_admissions'] + 1)
        
        # Momentum features (2nd derivative - acceleration)
        data['quantity_momentum'] = data['quantity_change'].diff().fillna(0)
        data['consumption_momentum'] = data['consumption_change'].diff().fillna(0)
        
        # Percentage change features
        data['quantity_pct_change'] = data['quantity'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        data['consumption_pct_change'] = data['consumption'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        
        # Trend direction indicator (-1, 0, or 1)
        data['trend_direction'] = np.sign(data['quantity_trend']).fillna(0)
        
        # Extract all 17 features in the correct order
        feature_cols = [
            # Base features (4)
            'quantity', 'consumption', 'resupply', 'total_admissions',
            # Trend features (4)
            'quantity_ma_7d', 'quantity_ma_14d', 'quantity_trend', 'consumption_trend',
            # Change features (2)
            'quantity_change', 'consumption_change',
            # Normalized features (2)
            'quantity_per_admission', 'consumption_rate',
            # Momentum features (2)
            'quantity_momentum', 'consumption_momentum',
            # Percentage change features (2)
            'quantity_pct_change', 'consumption_pct_change',
            # Directional indicator (1)
            'trend_direction'
        ]
        
        # Ensure all columns exist (fill missing with 0)
        for col in feature_cols:
            if col not in data.columns:
                data[col] = 0
        
        # Extract features and fill NaN values
        features = data[feature_cols].fillna(0).astype(float).values
        X = features[np.newaxis, :, :]  # Add batch dimension: (1, sequence_length, 17)
        
        # Predict
        predictions, lower, upper = self.predict(X, return_confidence=True)
        
        # Format output
        forecast_dates = pd.date_range(
            start=data['date'].max() + pd.Timedelta(days=1),
            periods=min(days_ahead, self.config['forecast_horizon'])
        )
        
        result = {
            'hospital_id': hospital_id,
            'resource_type': self.resource_type,
            'forecast_dates': forecast_dates.tolist(),
            'predicted_demand': predictions[0][:days_ahead].tolist(),
            'confidence_lower': lower[0][:days_ahead].tolist(),
            'confidence_upper': upper[0][:days_ahead].tolist(),
            'current_stock': float(data['quantity'].iloc[-1]),
            'avg_daily_consumption': float(data['consumption'].mean())
        }
        
        return result
    
    def save(self):
        """Save model and scalers"""
        self.model.save(str(self.model_path / 'model.h5'))
        joblib.dump(self.scaler, str(self.model_path / 'scaler.pkl'))
        joblib.dump(self.y_scaler, str(self.model_path / 'y_scaler.pkl'))  # NEW
        
        # Save config
        config = {
            'resource_type': self.resource_type,
            'sequence_length': self.config['sequence_length'],
            'forecast_horizon': self.config['forecast_horizon'],
            'input_features': [
                # Base features (4)
                'quantity', 'consumption', 'resupply', 'total_admissions',
                # Trend features (4)
                'quantity_ma_7d', 'quantity_ma_14d', 'quantity_trend', 'consumption_trend',
                # Change features (2)
                'quantity_change', 'consumption_change',
                # Normalized features (2)
                'quantity_per_admission', 'consumption_rate',
                # Momentum features (2)
                'quantity_momentum', 'consumption_momentum',
                # Percentage change features (2)
                'quantity_pct_change', 'consumption_pct_change',
                # Directional indicator (1)
                'trend_direction'
            ],
            'total_features': 17,
            'improvements': {
                'directional_loss': True,
                'momentum_features': True,
                'version': '2.0'
            }
        }
        with open(self.model_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model saved to {self.model_path}")
    
    def load(self):
        """Load trained model and scalers"""
        model_file = self.model_path / 'model.h5'
        scaler_file = self.model_path / 'scaler.pkl'
        y_scaler_file = self.model_path / 'y_scaler.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found at {model_file}")
        
        # Define custom objects for loading models with custom loss/metric functions
        # Keras saves custom functions by their name, so we need to provide them here
        # Create the loss function instance that matches what was used during training
        directional_loss_fn = create_directional_loss(magnitude_weight=0.7, direction_weight=0.3)
        
        # Provide multiple possible names that Keras might have used
        custom_objects = {
            'directional_loss': directional_loss_fn,
            'create_directional_loss': create_directional_loss,  # Factory function
            'directional_accuracy_metric': directional_accuracy_metric
        }
        
        try:
            # Try loading with custom objects first (for models with custom loss)
            self.model = keras.models.load_model(
                str(model_file),
                custom_objects=custom_objects,
                compile=False  # Load without compiling first
            )
            print("✓ Model loaded with custom loss function")
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # If custom objects don't work, try loading without compiling
            # This might happen for older models that used standard loss functions
            # or if the loss function name doesn't match
            print(f"⚠ Warning: Could not load with custom objects: {e}")
            print("  Attempting to load model without custom objects...")
            try:
                # Load model without compiling (avoids loss function resolution)
                self.model = keras.models.load_model(
                    str(model_file),
                    compile=False
                )
                # For inference, we don't need to recompile
                # The model weights are loaded, which is what matters for predictions
                print("✓ Model loaded (weights only, not compiled)")
            except Exception as e2:
                # Last resort: try loading and let Keras handle errors
                print(f"⚠ Trying direct load (may fail if custom loss needed)...")
                try:
                    self.model = keras.models.load_model(str(model_file))
                    print("✓ Model loaded (with default compilation)")
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to load model after multiple attempts.\n"
                        f"Last error: {e3}\n"
                        f"This model may have been trained with a custom loss function that can't be resolved.\n"
                        f"Consider retraining the model."
                    )
        
        self.scaler = joblib.load(str(scaler_file))
        
        # Load y_scaler if it exists (for backwards compatibility)
        if y_scaler_file.exists():
            self.y_scaler = joblib.load(str(y_scaler_file))
        else:
            # For old models without y_scaler, create a dummy scaler
            # (This means old models won't work correctly - will need retraining)
            self.y_scaler = StandardScaler()
            print(f"⚠ Warning: y_scaler not found. Old model may need retraining.")
        
        print(f"✓ Model loaded from {self.model_path}")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model performance
        
        Returns:
            Dict with evaluation metrics
        """
        # Predict (predict method handles X scaling and y inverse transform)
        # Pass X_test in original scale - predict() will scale it internally
        predictions = self.predict(X_test, return_confidence=False)
        
        # Convert to numpy arrays to ensure compatibility
        # This fixes the "return arrays must be of ArrayType" error
        y_test = np.asarray(y_test, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        
        # Fixed MAPE: Use symmetric MAPE to handle zero and small values
        # This prevents explosion when y_test is near zero
        epsilon = np.maximum(np.maximum(np.abs(y_test), np.abs(predictions)), 1e-6)
        mape = np.mean(np.abs(y_test - predictions) / epsilon) * 100
        
        # Clip MAPE to reasonable maximum (1000%) to prevent outliers from dominating
        mape = min(mape, 1000.0)
        
        # Directional accuracy (improved calculation)
        # Calculate differences between consecutive days
        # Ensure arrays are properly shaped for diff operation
        y_diff = np.diff(y_test, axis=1)
        pred_diff = np.diff(predictions, axis=1)
        
        # Convert to float64 to ensure compatibility
        y_diff = np.asarray(y_diff, dtype=np.float64)
        pred_diff = np.asarray(pred_diff, dtype=np.float64)
        
        # Filter out near-zero differences (no significant direction change)
        # Use percentile threshold to ignore noise-level changes
        abs_y_diff = np.abs(y_diff)
        threshold = np.percentile(abs_y_diff, 10) if len(abs_y_diff) > 0 else 0.01
        
        # Only consider significant changes (either actual or predicted)
        significant_mask = (abs_y_diff > threshold) | (np.abs(pred_diff) > threshold)
        
        if np.sum(significant_mask) > 0:
            # Calculate directional accuracy only for significant changes
            directional_accuracy = np.mean(
                (y_diff[significant_mask] * pred_diff[significant_mask]) > 0
            )
        else:
            # No significant changes detected
            directional_accuracy = 0.5  # Default to random
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy),
            'resource_type': self.resource_type
        }
        
        return metrics