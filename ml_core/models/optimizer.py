"""
Optimization Engine - Linear Programming
Generates optimal resource allocation strategies using mathematical optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pulp import *
import json
from datetime import datetime

# Handle both relative and absolute imports
try:
    from ..config import OPTIMIZATION_CONFIG, RESOURCE_TYPES
except ImportError:
    from config import OPTIMIZATION_CONFIG, RESOURCE_TYPES


class ResourceOptimizer:
    """Linear Programming-based resource allocation optimizer"""
    
    def __init__(self):
        self.config = OPTIMIZATION_CONFIG
        self.solver = None
    
    def calculate_transfer_cost(
        self,
        resource_type: str,
        quantity: int,
        distance_km: float
    ) -> float:
        """Calculate cost of transferring resources"""
        base_cost = self.config['transfer_cost_per_unit'].get(resource_type, 100)
        
        # Cost = base_cost * quantity + distance factor
        distance_factor = 1 + (distance_km / 100)  # Increases with distance
        total_cost = base_cost * quantity * distance_factor
        
        return total_cost
    
    def calculate_distance(
        self,
        hospital1: Dict,
        hospital2: Dict
    ) -> float:
        """Calculate distance between hospitals using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1 = hospital1['latitude'], hospital1['longitude']
        lat2, lon2 = hospital2['latitude'], hospital2['longitude']
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c  # Radius of Earth in kilometers
        
        return km
    
    def optimize_allocation(
        self,
        shortage_hospitals: pd.DataFrame,
        surplus_hospitals: pd.DataFrame,
        hospital_info: pd.DataFrame,
        resource_type: str,
        objective_weights: Dict[str, float] = None
    ) -> Dict:
        """
        Optimize resource allocation using Linear Programming
        
        Args:
            shortage_hospitals: DataFrame with hospitals needing resources
            surplus_hospitals: DataFrame with hospitals having excess
            hospital_info: Hospital metadata (location, capacity, etc.)
            resource_type: Type of resource to allocate
            objective_weights: Custom weights for objectives
        
        Returns:
            Dict with optimal allocation strategy
        """
        if objective_weights is None:
            objective_weights = self.config['objectives']
        
        # Create optimization problem
        prob = LpProblem("Resource_Allocation", LpMinimize)
        
        # Decision variables: x[i,j] = quantity to transfer from surplus i to shortage j
        transfers = {}
        for i, surplus in surplus_hospitals.iterrows():
            for j, shortage in shortage_hospitals.iterrows():
                surplus_id = surplus['hospital_id']
                shortage_id = shortage['hospital_id']
                
                # Skip if same hospital
                if surplus_id == shortage_id:
                    continue
                
                # Get distance
                # Handle both 'id' and 'hospital_id' column names
                id_col = 'hospital_id' if 'hospital_id' in hospital_info.columns else 'id'
                surplus_info = hospital_info[hospital_info[id_col] == surplus_id].iloc[0]
                shortage_info = hospital_info[hospital_info[id_col] == shortage_id].iloc[0]
                distance = self.calculate_distance(surplus_info.to_dict(), shortage_info.to_dict())
                
                # Skip if too far
                if distance > self.config['max_transfer_distance_km']:
                    continue
                
                # Create variable
                max_transfer = min(
                    surplus['available_quantity'],
                    shortage['quantity_needed']
                )
                
                if max_transfer > 0:
                    var_name = f"transfer_{surplus_id[:8]}_{shortage_id[:8]}"
                    transfers[(surplus_id, shortage_id)] = {
                        'var': LpVariable(var_name, 0, max_transfer, LpInteger),
                        'cost': self.calculate_transfer_cost(resource_type, 1, distance),
                        'distance': distance,
                        'max_quantity': max_transfer
                    }
        
        if not transfers:
            return {
                'status': 'no_feasible_transfers',
                'resource_type': resource_type,
                'message': 'No surplus hospitals within range of shortage hospitals',
                'allocations': []
            }
        
        # Objective function components
        
        # 1. Minimize total cost
        cost_objective = lpSum([
            info['var'] * info['cost']
            for info in transfers.values()
        ])
        
        # 2. Minimize unmet shortage (shortage penalty)
        shortage_penalties = {}
        for j, shortage in shortage_hospitals.iterrows():
            shortage_id = shortage['hospital_id']
            quantity_needed = shortage['quantity_needed']
            
            # Calculate how much this shortage hospital receives
            received = lpSum([
                transfers[(surplus_id, shortage_id)]['var']
                for surplus_id, sid in transfers.keys()
                if sid == shortage_id
            ])
            
            # Unmet shortage
            unmet = quantity_needed - received
            shortage_penalties[shortage_id] = unmet * 10000  # High penalty for unmet needs
        
        shortage_objective = lpSum(shortage_penalties.values())
        
        # 3. Minimize number of transfers (complexity)
        # Binary variables to track if transfer happens
        transfer_binary = {}
        M = 1000000  # Large constant for big-M formulation
        for key in transfers.keys():
            var_name = f"binary_{key[0][:8]}_{key[1][:8]}"
            transfer_binary[key] = LpVariable(var_name, 0, 1, LpBinary)
            
            # Link binary to transfer: if transfer > 0, binary = 1
            # Big-M formulation: transfer <= M * binary (if transfer > 0, binary must be 1)
            # Also: transfer >= 0.01 * binary (if binary = 1, transfer must be > 0, but LP needs >)
            prob += transfers[key]['var'] <= M * transfer_binary[key]
            prob += transfers[key]['var'] >= 0.01 * transfer_binary[key]  # If binary=1, transfer must be at least 0.01
        
        complexity_objective = lpSum(transfer_binary.values()) * 100
        
        # Combined objective
        prob += (
            objective_weights['minimize_shortage'] * shortage_objective +
            objective_weights['minimize_cost'] * cost_objective / 10000 +  # Normalize
            objective_weights['maximize_coverage'] * complexity_objective
        )
        
        # Constraints
        
        # 1. Don't exceed available surplus at each source
        for surplus_id in surplus_hospitals['hospital_id']:
            available = surplus_hospitals[
                surplus_hospitals['hospital_id'] == surplus_id
            ]['available_quantity'].values[0]
            
            prob += lpSum([
                transfers[(sid, shortage_id)]['var']
                for sid, shortage_id in transfers.keys()
                if sid == surplus_id
            ]) <= available
        
        # 2. Don't over-allocate to shortage hospitals
        for shortage_id in shortage_hospitals['hospital_id']:
            needed = shortage_hospitals[
                shortage_hospitals['hospital_id'] == shortage_id
            ]['quantity_needed'].values[0]
            
            prob += lpSum([
                transfers[(surplus_id, sid)]['var']
                for surplus_id, sid in transfers.keys()
                if sid == shortage_id
            ]) <= needed
        
        # 3. Fairness constraint: ensure all critical hospitals get at least something
        critical_hospitals = shortage_hospitals[
            shortage_hospitals['risk_level'] == 'critical'
        ]
        for _, critical in critical_hospitals.iterrows():
            critical_id = critical['hospital_id']
            
            # Check if there are any feasible transfers to this critical hospital
            critical_transfers = [
                key for key in transfers.keys()
                if key[1] == critical_id
            ]
            
            if critical_transfers:
                # Must receive at least 50% of need or 1 unit (whichever is less)
                # But don't force if it would make problem infeasible
                min_allocation = min(1, int(critical['quantity_needed'] * 0.5))
                if min_allocation > 0:
                    prob += lpSum([
                        transfers[key]['var']
                        for key in critical_transfers
                    ]) >= min_allocation
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=self.config['time_limit']))
        
        # Extract solution
        status = LpStatus[prob.status]
        
        if status != 'Optimal':
            return {
                'status': status.lower() if isinstance(status, str) else status,
                'resource_type': resource_type,
                'message': f'Optimization failed with status: {status}',
                'allocations': []
            }
        
        # Build allocation list
        allocations = []
        total_cost = 0
        total_transferred = 0
        
        for (surplus_id, shortage_id), info in transfers.items():
            quantity = int(value(info['var']))
            
            if quantity > 0:
                cost = quantity * info['cost']
                total_cost += cost
                total_transferred += quantity
                
                allocations.append({
                    'from_hospital_id': surplus_id,
                    'to_hospital_id': shortage_id,
                    'resource_type': resource_type,
                    'quantity': quantity,
                    'transfer_cost': round(cost, 2),
                    'distance_km': round(info['distance'], 2),
                    'estimated_time_hours': int(info['distance'] / 60)  # Assume 60 km/h
                })
        
        # Calculate metrics
        total_shortage_before = shortage_hospitals['quantity_needed'].sum()
        total_shortage_after = total_shortage_before - total_transferred
        shortage_reduction = (total_transferred / total_shortage_before) * 100
        
        # Count hospitals helped
        hospitals_helped = len(set([a['to_hospital_id'] for a in allocations]))
        
        result = {
            'status': 'optimal',
            'resource_type': resource_type,
            'allocations': allocations,
            'summary': {
                'total_transfers': len(allocations),
                'total_quantity_transferred': total_transferred,
                'total_cost': round(total_cost, 2),
                'hospitals_helped': hospitals_helped,
                'shortage_reduction_percent': round(shortage_reduction, 2),
                'total_shortage_before': int(total_shortage_before),
                'total_shortage_after': int(total_shortage_after),
                'objective_value': round(value(prob.objective), 2)
            }
        }
        
        return result
    
    def generate_multiple_strategies(
        self,
        shortage_hospitals: pd.DataFrame,
        surplus_hospitals: pd.DataFrame,
        hospital_info: pd.DataFrame,
        resource_type: str,
        n_strategies: int = 3
    ) -> List[Dict]:
        """
        Generate multiple allocation strategies with different objective weights
        
        Returns:
            List of strategies ranked by different criteria
        """
        strategies = []
        
        # Strategy 1: Minimize cost (cost-efficient)
        strategy1 = self.optimize_allocation(
            shortage_hospitals, surplus_hospitals, hospital_info, resource_type,
            objective_weights={
                'minimize_shortage': 0.5,
                'minimize_cost': 1.0,
                'maximize_coverage': 0.3,
                'fairness': 0.3
            }
        )
        if strategy1['status'] == 'optimal':
            strategy1['strategy_name'] = 'Cost-Efficient'
            strategy1['strategy_description'] = 'Minimizes transfer costs while addressing critical shortages'
            strategy1['cost_score'] = 1.0
            strategy1['speed_score'] = 0.6
            strategy1['coverage_score'] = 0.7
            strategies.append(strategy1)
        
        # Strategy 2: Maximize coverage (help most hospitals)
        strategy2 = self.optimize_allocation(
            shortage_hospitals, surplus_hospitals, hospital_info, resource_type,
            objective_weights={
                'minimize_shortage': 1.0,
                'minimize_cost': 0.3,
                'maximize_coverage': 1.0,
                'fairness': 0.8
            }
        )
        if strategy2['status'] == 'optimal':
            strategy2['strategy_name'] = 'Maximum Coverage'
            strategy2['strategy_description'] = 'Helps the maximum number of hospitals'
            strategy2['cost_score'] = 0.6
            strategy2['speed_score'] = 0.5
            strategy2['coverage_score'] = 1.0
            strategies.append(strategy2)
        
        # Strategy 3: Balanced approach
        strategy3 = self.optimize_allocation(
            shortage_hospitals, surplus_hospitals, hospital_info, resource_type,
            objective_weights={
                'minimize_shortage': 0.8,
                'minimize_cost': 0.6,
                'maximize_coverage': 0.7,
                'fairness': 0.7
            }
        )
        if strategy3['status'] == 'optimal':
            strategy3['strategy_name'] = 'Balanced'
            strategy3['strategy_description'] = 'Balanced approach considering cost, coverage, and urgency'
            strategy3['cost_score'] = 0.75
            strategy3['speed_score'] = 0.75
            strategy3['coverage_score'] = 0.85
            strategies.append(strategy3)
        
        # Rank by overall score
        for strategy in strategies:
            strategy['overall_score'] = (
                strategy['cost_score'] * 0.3 +
                strategy['speed_score'] * 0.3 +
                strategy['coverage_score'] * 0.4
            )
        
        strategies.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return strategies
    
    def validate_allocation(
        self,
        allocation: Dict,
        current_inventory: pd.DataFrame
    ) -> Dict:
        """
        Validate that an allocation is feasible given current inventory
        
        Returns:
            Dict with validation result and issues (if any)
        """
        issues = []
        
        for transfer in allocation['allocations']:
            from_id = transfer['from_hospital_id']
            quantity = transfer['quantity']
            
            # Check if source has enough
            source_inventory = current_inventory[
                current_inventory['hospital_id'] == from_id
            ]
            
            if source_inventory.empty:
                issues.append(f"Source hospital {from_id} not found in inventory")
                continue
            
            available = source_inventory['available_quantity'].values[0]
            
            if quantity > available:
                issues.append(
                    f"Transfer from {from_id} requests {quantity} but only {available} available"
                )
        
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }