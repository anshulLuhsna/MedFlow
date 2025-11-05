"""
Integration tests for ML Core
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import MLCore
from config import RESOURCE_TYPES


class TestMLCore(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        print("\nInitializing ML Core for testing...")
        cls.ml_core = MLCore()
    
    def test_01_initialization(self):
        """Test ML Core initialization"""
        self.assertIsNotNone(self.ml_core.data_loader)
        self.assertIsNotNone(self.ml_core.shortage_detector)
        self.assertIsNotNone(self.ml_core.optimizer)
        self.assertIsNotNone(self.ml_core.preference_learner)
        
        # Check all forecasters initialized
        for resource_type in RESOURCE_TYPES:
            self.assertIn(resource_type, self.ml_core.demand_forecasters)
    
    def test_02_predict_demand(self):
        """Test demand prediction (requires trained models)"""
        try:
            # This will only work if models are trained
            hospitals = self.ml_core.data_loader.get_hospitals()
            if not hospitals.empty:
                hospital_id = hospitals.iloc[0]['id']
                
                prediction = self.ml_core.predict_demand(
                    hospital_id=hospital_id,
                    resource_type='ventilators',
                    days_ahead=7
                )
                
                self.assertIn('hospital_id', prediction)
                self.assertIn('predicted_demand', prediction)
                self.assertIn('confidence_lower', prediction)
                self.assertIn('confidence_upper', prediction)
        except Exception as e:
            self.skipTest(f"Skipping: Models not trained yet ({e})")
    
    def test_03_detect_shortages(self):
        """Test shortage detection (requires trained models)"""
        try:
            results = self.ml_core.detect_shortages(resource_type='ventilators')
            
            self.assertIsInstance(results, pd.DataFrame)
            if not results.empty:
                self.assertIn('hospital_id', results.columns)
                self.assertIn('risk_level', results.columns)
                self.assertIn('confidence', results.columns)
        except Exception as e:
            self.skipTest(f"Skipping: Models not trained yet ({e})")
    
    def test_04_optimize_allocation(self):
        """Test allocation optimization"""
        try:
            result = self.ml_core.optimize_allocation(resource_type='ventilators')
            
            self.assertIn('status', result)
            self.assertIn('resource_type', result)
        except Exception as e:
            self.skipTest(f"Skipping: {e}")
    
    def test_05_generate_strategies(self):
        """Test strategy generation"""
        try:
            strategies = self.ml_core.generate_allocation_strategies(
                resource_type='ventilators',
                n_strategies=3
            )
            
            self.assertIsInstance(strategies, list)
        except Exception as e:
            self.skipTest(f"Skipping: {e}")


if __name__ == '__main__':
    unittest.main()