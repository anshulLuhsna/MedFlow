"""
Tests for Demand Forecaster
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.demand_forecaster import DemandForecaster
from config import DEMAND_FORECAST_CONFIG


class TestDemandForecaster(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.resource_type = "ventilators"
        cls.forecaster = DemandForecaster(cls.resource_type)
        
        # Create synthetic test data
        cls.n_samples = 100
        cls.seq_len = DEMAND_FORECAST_CONFIG['sequence_length']
        cls.n_features = 4  # quantity, consumption, resupply, admissions
        
        cls.X_train = np.random.rand(cls.n_samples, cls.seq_len, cls.n_features) * 100
        cls.y_train = np.random.rand(cls.n_samples, 14) * 50
        
        cls.X_test = np.random.rand(20, cls.seq_len, cls.n_features) * 100
        cls.y_test = np.random.rand(20, 14) * 50
    
    def test_01_build_model(self):
        """Test model architecture building"""
        input_shape = (self.seq_len, self.n_features)
        model = self.forecaster.build_model(input_shape)
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 9)  # Check architecture
    
    def test_02_prepare_sequences(self):
        """Test data preparation and normalization"""
        X_scaled, y_scaled = self.forecaster.prepare_sequences(self.X_train, self.y_train)
        
        self.assertEqual(X_scaled.shape, self.X_train.shape)
        self.assertEqual(y_scaled.shape, self.y_train.shape)
    
    def test_03_train(self):
        """Test model training"""
        history = self.forecaster.train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_test,
            y_val=self.y_test,
            verbose=0
        )
        
        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
        self.assertTrue(len(history['loss']) > 0)
    
    def test_04_predict(self):
        """Test prediction"""
        predictions = self.forecaster.predict(self.X_test)
        
        self.assertEqual(predictions.shape, self.y_test.shape)
        self.assertTrue(np.all(predictions >= 0))  # Non-negative predictions
    
    def test_05_predict_with_confidence(self):
        """Test prediction with confidence intervals"""
        predictions, lower, upper = self.forecaster.predict(
            self.X_test,
            return_confidence=True
        )
        
        self.assertEqual(predictions.shape, self.y_test.shape)
        self.assertEqual(lower.shape, self.y_test.shape)
        self.assertEqual(upper.shape, self.y_test.shape)
        self.assertTrue(np.all(lower <= predictions))
        self.assertTrue(np.all(predictions <= upper))
    
    def test_06_evaluate(self):
        """Test model evaluation"""
        metrics = self.forecaster.evaluate(self.X_test, self.y_test)
        
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('directional_accuracy', metrics)
        
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
    
    def test_07_save_load(self):
        """Test model saving and loading"""
        # Save
        self.forecaster.save()
        
        # Create new forecaster and load
        new_forecaster = DemandForecaster(self.resource_type)
        new_forecaster.load()
        
        self.assertIsNotNone(new_forecaster.model)
        self.assertIsNotNone(new_forecaster.scaler)
        
        # Predictions should be similar
        pred1 = self.forecaster.predict(self.X_test[:5])
        pred2 = new_forecaster.predict(self.X_test[:5])
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=2)


if __name__ == '__main__':
    unittest.main()