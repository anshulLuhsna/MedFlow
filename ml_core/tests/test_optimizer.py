"""
Tests for Resource Optimizer
"""

import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.optimizer import ResourceOptimizer


class TestResourceOptimizer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.optimizer = ResourceOptimizer()
        
        # Create test data
        cls.shortage_hospitals = pd.DataFrame({
            'hospital_id': ['shortage_1', 'shortage_2', 'shortage_3'],
            'resource_type': ['ventilators', 'ventilators', 'ventilators'],
            'quantity_needed': [5, 10, 3],
            'current_stock': [1, 2, 0],
            'risk_level': ['critical', 'high', 'medium'],
            'available_quantity': [1, 2, 0]
        })
        
        cls.surplus_hospitals = pd.DataFrame({
            'hospital_id': ['surplus_1', 'surplus_2'],
            'resource_type': ['ventilators', 'ventilators'],
            'available_quantity': [15, 8],
            'current_stock': [20, 12]
        })
        
        cls.hospital_info = pd.DataFrame({
            'hospital_id': ['shortage_1', 'shortage_2', 'shortage_3', 'surplus_1', 'surplus_2'],
            'name': ['Hospital A', 'Hospital B', 'Hospital C', 'Hospital D', 'Hospital E'],
            'latitude': [40.7128, 34.0522, 41.8781, 42.3601, 39.7392],
            'longitude': [-74.0060, -118.2437, -87.6298, -71.0589, -104.9903],
            'region': ['North', 'South', 'Central', 'North', 'West'],
            'capacity_beds': [200, 300, 150, 400, 250]
        })
    
    def test_01_calculate_distance(self):
        """Test distance calculation"""
        hospital1 = {
            'latitude': 40.7128,
            'longitude': -74.0060
        }
        hospital2 = {
            'latitude': 34.0522,
            'longitude': -118.2437
        }
        
        distance = self.optimizer.calculate_distance(hospital1, hospital2)
        
        self.assertGreater(distance, 0)
        self.assertLess(distance, 5000)  # Should be reasonable
    
    def test_02_calculate_transfer_cost(self):
        """Test transfer cost calculation"""
        cost = self.optimizer.calculate_transfer_cost(
            resource_type='ventilators',
            quantity=5,
            distance_km=100
        )
        
        self.assertGreater(cost, 0)
        
        # Cost should increase with distance
        cost2 = self.optimizer.calculate_transfer_cost(
            resource_type='ventilators',
            quantity=5,
            distance_km=200
        )
        self.assertGreater(cost2, cost)
    
    def test_03_optimize_allocation(self):
        """Test optimization"""
        result = self.optimizer.optimize_allocation(
            shortage_hospitals=self.shortage_hospitals,
            surplus_hospitals=self.surplus_hospitals,
            hospital_info=self.hospital_info,
            resource_type='ventilators'
        )
        
        self.assertIn('status', result)
        
        if result['status'] == 'optimal':
            self.assertIn('allocations', result)
            self.assertIn('summary', result)
            
            # Check allocations are valid
            for allocation in result['allocations']:
                self.assertIn('from_hospital_id', allocation)
                self.assertIn('to_hospital_id', allocation)
                self.assertIn('quantity', allocation)
                self.assertGreater(allocation['quantity'], 0)
    
    def test_04_generate_multiple_strategies(self):
        """Test multiple strategy generation"""
        strategies = self.optimizer.generate_multiple_strategies(
            shortage_hospitals=self.shortage_hospitals,
            surplus_hospitals=self.surplus_hospitals,
            hospital_info=self.hospital_info,
            resource_type='ventilators',
            n_strategies=3
        )
        
        self.assertIsInstance(strategies, list)
        self.assertLessEqual(len(strategies), 3)
        
        # Check each strategy has required fields
        for strategy in strategies:
            self.assertIn('strategy_name', strategy)
            self.assertIn('allocations', strategy)
            self.assertIn('cost_score', strategy)
            self.assertIn('coverage_score', strategy)
    
    def test_05_validate_allocation(self):
        """Test allocation validation"""
        # First generate an allocation
        result = self.optimizer.optimize_allocation(
            shortage_hospitals=self.shortage_hospitals,
            surplus_hospitals=self.surplus_hospitals,
            hospital_info=self.hospital_info,
            resource_type='ventilators'
        )
        
        if result['status'] == 'optimal':
            # Create current inventory
            current_inventory = pd.concat([
                self.shortage_hospitals[['hospital_id', 'available_quantity']],
                self.surplus_hospitals[['hospital_id', 'available_quantity']]
            ])
            
            validation = self.optimizer.validate_allocation(result, current_inventory)
            
            self.assertIn('is_valid', validation)
            self.assertIn('issues', validation)
            self.assertIsInstance(validation['is_valid'], bool)
    
    def test_06_empty_shortages(self):
        """Test with empty shortage hospitals"""
        empty_shortages = pd.DataFrame(columns=['hospital_id', 'quantity_needed', 'risk_level'])
        
        result = self.optimizer.optimize_allocation(
            shortage_hospitals=empty_shortages,
            surplus_hospitals=self.surplus_hospitals,
            hospital_info=self.hospital_info,
            resource_type='ventilators'
        )
        
        self.assertIn('status', result)
        self.assertEqual(len(result['allocations']), 0)
    
    def test_07_empty_surpluses(self):
        """Test with empty surplus hospitals"""
        empty_surpluses = pd.DataFrame(columns=['hospital_id', 'available_quantity'])
        
        result = self.optimizer.optimize_allocation(
            shortage_hospitals=self.shortage_hospitals,
            surplus_hospitals=empty_surpluses,
            hospital_info=self.hospital_info,
            resource_type='ventilators'
        )
        
        self.assertIn('status', result)
        # Should return no_feasible_transfers or similar
        self.assertIn(result['status'], ['no_feasible_transfers', 'optimal'])
    
    def test_08_all_too_far(self):
        """Test when all hospitals are beyond max_transfer_distance"""
        # Create hospitals very far apart (>200km)
        far_hospitals = pd.DataFrame({
            'hospital_id': ['shortage_1', 'surplus_1'],
            'name': ['Hospital A', 'Hospital B'],
            'latitude': [37.7749, 40.0],  # ~250km apart
            'longitude': [-122.4194, -122.4194],
            'region': ['North', 'South'],
            'capacity_beds': [200, 300]
        })
        
        result = self.optimizer.optimize_allocation(
            shortage_hospitals=self.shortage_hospitals.head(1),  # Just first shortage
            surplus_hospitals=self.surplus_hospitals.head(1),  # Just first surplus
            hospital_info=far_hospitals,
            resource_type='ventilators'
        )
        
        self.assertIn('status', result)
        # Should handle gracefully (either no_feasible_transfers or skip far hospitals)
    
    def test_09_different_resource_types(self):
        """Test optimization for different resource types"""
        resource_types = ['ppe', 'o2_cylinders', 'medications']
        
        for resource_type in resource_types:
            result = self.optimizer.optimize_allocation(
                shortage_hospitals=self.shortage_hospitals,
                surplus_hospitals=self.surplus_hospitals,
                hospital_info=self.hospital_info,
                resource_type=resource_type
            )
            
            self.assertIn('status', result)
            # resource_type is only set when status is 'optimal'
            if result['status'] == 'optimal':
                self.assertEqual(result.get('resource_type'), resource_type)


if __name__ == '__main__':
    unittest.main()