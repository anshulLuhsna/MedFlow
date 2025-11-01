"""
Demand Forecasting Model - LSTM Neural Network
Predicts future resource demand based on historical patterns
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


class DemandForecaster:
    """LSTM-based demand forecasting model"""
    
    def __init__(self, resource_type: str):
        if resource_type not in RESOURCE_TYPES:
            raise ValueError(f"Invalid resource type: {resource_type}")
        
        self.resource_type = resource_type
        self.config = DEMAND_FORECAST_CONFIG
        self.model = None
        self.scaler = StandardScaler()
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
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.config['dropout']),
            layers.Dense(32, activation='relu'),
            
            # Output layer (predict next forecast_horizon days)
            layers.Dense(self.config['forecast_horizon'])
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        
        # Use Huber loss for better robustness and directional learning
        # Huber loss is less sensitive to outliers than MSE
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust than MSE, helps with directional accuracy
            metrics=['mae', 'mape']
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
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        
        return X_scaled, y
    
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
        # Prepare data
        X_train_scaled, y_train_scaled = self.prepare_sequences(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(
                X_val.reshape(-1, X_val.shape[2])
            ).reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
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
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
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
        return_confidence: bool = False
    ) -> np.ndarray:
        """
        Predict future demand
        
        Args:
            X: Input sequences (samples, sequence_length, features)
            return_confidence: Whether to return prediction intervals
        
        Returns:
            Predictions (samples, forecast_horizon)
            If return_confidence=True: (predictions, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Scale input
        n_samples, seq_len, n_features = X.shape
        X_scaled = self.scaler.transform(
            X.reshape(-1, n_features)
        ).reshape(n_samples, seq_len, n_features)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        if return_confidence:
            # Simple confidence intervals (can be improved with MC Dropout)
            std = np.std(predictions, axis=0, keepdims=True)
            lower = predictions - 1.96 * std
            upper = predictions + 1.96 * std
            return predictions, lower, upper
        
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
            historical_data: DataFrame with columns [date, quantity, consumption, admissions]
            days_ahead: Number of days to forecast
        
        Returns:
            Dict with predictions and metadata
        """
        # Prepare sequence
        data = historical_data.sort_values('date').tail(self.config['sequence_length'])
        
        if len(data) < self.config['sequence_length']:
            raise ValueError(f"Need at least {self.config['sequence_length']} days of history")
        
        features = data[['quantity', 'consumption', 'resupply', 'total_admissions']].fillna(0).values
        X = features[np.newaxis, :, :]  # Add batch dimension
        
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
        """Save model and scaler"""
        self.model.save(str(self.model_path / 'model.h5'))
        joblib.dump(self.scaler, str(self.model_path / 'scaler.pkl'))
        
        # Save config
        config = {
            'resource_type': self.resource_type,
            'sequence_length': self.config['sequence_length'],
            'forecast_horizon': self.config['forecast_horizon'],
            'input_features': ['quantity', 'consumption', 'resupply', 'total_admissions']
        }
        with open(self.model_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model saved to {self.model_path}")
    
    def load(self):
        """Load trained model and scaler"""
        model_file = self.model_path / 'model.h5'
        scaler_file = self.model_path / 'scaler.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found at {model_file}")
        
        self.model = keras.models.load_model(str(model_file))
        self.scaler = joblib.load(str(scaler_file))
        
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
        # Scale data
        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, X_test.shape[2])
        ).reshape(X_test.shape)
        
        # Predict
        predictions = self.model.predict(X_test_scaled, verbose=0)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
        
        # Directional accuracy (did we predict increase/decrease correctly?)
        y_diff = np.diff(y_test, axis=1)
        pred_diff = np.diff(predictions, axis=1)
        directional_accuracy = np.mean((y_diff * pred_diff) > 0)
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy),
            'resource_type': self.resource_type
        }
        
        return metrics