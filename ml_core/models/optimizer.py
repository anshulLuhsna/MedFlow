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
        objective_weights: Dict[str, float] = None,
        strategy_name: str = None
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
        skipped_same_hospital = 0
        skipped_distance = 0
        skipped_zero_quantity = 0
        
        print(f"[Optimizer] Creating transfer variables: {len(surplus_hospitals)} surplus × {len(shortage_hospitals)} shortage = {len(surplus_hospitals) * len(shortage_hospitals)} potential pairs")
        
        for i, surplus in surplus_hospitals.iterrows():
            for j, shortage in shortage_hospitals.iterrows():
                surplus_id = surplus['hospital_id']
                shortage_id = shortage['hospital_id']
                
                # Skip if same hospital
                if surplus_id == shortage_id:
                    skipped_same_hospital += 1
                    continue
                
                # Get distance
                # Handle both 'id' and 'hospital_id' column names
                id_col = 'hospital_id' if 'hospital_id' in hospital_info.columns else 'id'
                surplus_info = hospital_info[hospital_info[id_col] == surplus_id].iloc[0]
                shortage_info = hospital_info[hospital_info[id_col] == shortage_id].iloc[0]
                distance = self.calculate_distance(surplus_info.to_dict(), shortage_info.to_dict())
                
                # Skip if too far
                if distance > self.config['max_transfer_distance_km']:
                    skipped_distance += 1
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
                else:
                    skipped_zero_quantity += 1
        
        print(f"[Optimizer] Created {len(transfers)} transfer variables")
        print(f"[Optimizer] Skipped: {skipped_same_hospital} same hospital, {skipped_distance} distance, {skipped_zero_quantity} zero quantity")
        
        if not transfers:
            return {
                'status': 'no_feasible_transfers',
                'resource_type': resource_type,
                'message': f'No feasible transfers found. Skipped: {skipped_same_hospital} same hospital, {skipped_distance} distance > {self.config["max_transfer_distance_km"]}km, {skipped_zero_quantity} zero quantity',
                'allocations': []
            }
        
        # Objective function components
        
        # 1. Minimize total cost
        cost_objective = lpSum([
            info['var'] * info['cost']
            for info in transfers.values()
        ])
        
        # 2. Minimize unmet shortage (shortage penalty) - REDUCED DOMINANCE
        shortage_penalties = {}
        for j, shortage in shortage_hospitals.iterrows():
            shortage_id = shortage['hospital_id']
            quantity_needed = shortage['quantity_needed']
            risk_level = shortage.get('risk_level', 'medium')
            
            # Calculate how much this shortage hospital receives
            received = lpSum([
                transfers[(surplus_id, shortage_id)]['var']
                for surplus_id, sid in transfers.keys()
                if sid == shortage_id
            ])
            
            # Unmet shortage
            unmet = quantity_needed - received
            
            # Penalty proportional to risk level (reduced from 10000x)
            if risk_level == 'critical':
                penalty_multiplier = 1000  # High but not overwhelming
            elif risk_level == 'high':
                penalty_multiplier = 500
            else:
                penalty_multiplier = 100  # Low/medium risk
            
            shortage_penalties[shortage_id] = unmet * penalty_multiplier
        
        shortage_objective = lpSum(shortage_penalties.values())
        
        # 3. Maximize number of hospitals helped (coverage) - FIXED
        # Binary variables to track if each shortage hospital receives ANY aid
        hospitals_helped_binary = {}
        M = 1000000  # Large constant for big-M formulation
        
        for shortage_id in shortage_hospitals['hospital_id']:
            var_name = f"helped_{shortage_id[:8]}"
            hospitals_helped_binary[shortage_id] = LpVariable(var_name, 0, 1, LpBinary)
            
            # Calculate total received by this hospital
            received = lpSum([
                transfers[(surplus_id, sid)]['var']
                for surplus_id, sid in transfers.keys()
                if sid == shortage_id
            ])
            
            # Link binary to transfers: if received > 0, binary = 1
            # Big-M formulation: received <= M * binary (if received > 0, binary must be 1)
            # Also: received >= 0.01 * binary (if binary = 1, received must be > 0)
            prob += received <= M * hospitals_helped_binary[shortage_id]
            prob += received >= 0.01 * hospitals_helped_binary[shortage_id]
        
        # Coverage objective: maximize hospitals helped (negative because we're minimizing)
        coverage_objective = -lpSum(hospitals_helped_binary.values()) * 1000
        
        # 4. Minimize number of transfers (complexity/fairness)
        # Binary variables to track if transfer happens
        transfer_binary = {}
        for key in transfers.keys():
            var_name = f"binary_{key[0][:8]}_{key[1][:8]}"
            transfer_binary[key] = LpVariable(var_name, 0, 1, LpBinary)
            
            # Link binary to transfer: if transfer > 0, binary = 1
            prob += transfers[key]['var'] <= M * transfer_binary[key]
            prob += transfers[key]['var'] >= 0.01 * transfer_binary[key]
        
        complexity_objective = lpSum(transfer_binary.values()) * 100
        
        # Combined objective
        prob += (
            objective_weights['minimize_shortage'] * shortage_objective +
            objective_weights['minimize_cost'] * cost_objective / 10000 +  # Normalize
            objective_weights['maximize_coverage'] * coverage_objective +  # FIXED: now maximizes hospitals helped
            objective_weights.get('fairness', 0.5) * complexity_objective  # Fairness = fewer transfers
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
        
        # 4. Strategy-specific constraints to force different solutions
        # Use adaptive constraints: if too restrictive, relax them
        constraint_applied = False
        if strategy_name == 'Cost-Efficient':
            # Cost-Efficient: Limit total cost to force cheaper solutions
            total_shortage = shortage_hospitals['quantity_needed'].sum()
            # Estimate cost: use actual transfer costs from available transfers
            if transfers:
                avg_cost_per_unit = sum(info['cost'] for info in transfers.values()) / len(transfers)
            else:
                avg_cost_per_unit = 1500
            # Start with 70% budget, but will relax if infeasible
            cost_budget = total_shortage * avg_cost_per_unit * 0.70
            prob += cost_objective <= cost_budget
            constraint_applied = True
            print(f"[Optimizer] Cost-Efficient: Added cost budget constraint: ${cost_budget:.2f}")
        
        elif strategy_name == 'Maximum Coverage':
            # Maximum Coverage: Must help at least 80% of hospitals (relaxed from 100%)
            # Requiring 100% might be infeasible if not enough resources
            min_hospitals_to_help = max(1, int(len(shortage_hospitals) * 0.8))
            prob += lpSum(hospitals_helped_binary.values()) >= min_hospitals_to_help
            constraint_applied = True
            print(f"[Optimizer] Maximum Coverage: Must help at least {min_hospitals_to_help} of {len(shortage_hospitals)} hospitals")
        
        elif strategy_name == 'Balanced':
            # Balanced: Limit maximum single transfer size for fairness
            # Use adaptive limit: 60% of max possible transfer
            if transfers:
                max_possible = max(info['max_quantity'] for info in transfers.values())
                max_single_transfer = max(3, int(max_possible * 0.6))  # At least 3, but 60% of max
            else:
                max_single_transfer = 5
            for key in transfers.keys():
                prob += transfers[key]['var'] <= max_single_transfer
            constraint_applied = True
            print(f"[Optimizer] Balanced: Limited max transfer size to {max_single_transfer} units")
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=self.config['time_limit']))
        
        # Extract solution
        status = LpStatus[prob.status]
        
        if status != 'Optimal':
            # If infeasible due to constraints, try relaxing them
            if strategy_name and status == 'Infeasible':
                print(f"[Optimizer] WARNING: Strategy '{strategy_name}' resulted in infeasible solution. Status: {status}")
                print(f"[Optimizer] This may happen if constraints are too restrictive. Consider adjusting constraints.")
            
            return {
                'status': status.lower() if isinstance(status, str) else status,
                'resource_type': resource_type,
                'message': f'Optimization failed with status: {status}. Strategy: {strategy_name or "default"}',
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
        
        # Strategy 1: Cost-Efficient - DRAMATICALLY prioritize cost minimization
        strategy1 = self.optimize_allocation(
            shortage_hospitals, surplus_hospitals, hospital_info, resource_type,
            objective_weights={
                'minimize_shortage': 0.3,   # Low: willing to leave some shortages if expensive
                'minimize_cost': 3.0,       # HIGH: cost is priority
                'maximize_coverage': 0.1,   # Low: don't care about number of hospitals
                'fairness': 0.5             # Medium: prefer fewer, larger transfers
            },
            strategy_name='Cost-Efficient'
        )
        if strategy1['status'] == 'optimal':
            strategy1['strategy_name'] = 'Cost-Efficient'
            strategy1['strategy_description'] = 'Minimizes transfer costs while addressing critical shortages'
            # Calculate scores based on actual performance
            summary = strategy1.get('summary', {})
            strategies.append(strategy1)
        
        # Strategy 2: Maximum Coverage - DRAMATICALLY prioritize helping most hospitals
        strategy2 = self.optimize_allocation(
            shortage_hospitals, surplus_hospitals, hospital_info, resource_type,
            objective_weights={
                'minimize_shortage': 0.5,   # Medium: address shortages
                'minimize_cost': 0.1,       # Low: cost is not priority
                'maximize_coverage': 3.0,   # HIGH: help as many hospitals as possible
                'fairness': 0.2             # Low: ok with many small transfers
            },
            strategy_name='Maximum Coverage'
        )
        if strategy2['status'] == 'optimal':
            strategy2['strategy_name'] = 'Maximum Coverage'
            strategy2['strategy_description'] = 'Helps the maximum number of hospitals'
            strategies.append(strategy2)
        
        # Strategy 3: Balanced - Equal priorities
        strategy3 = self.optimize_allocation(
            shortage_hospitals, surplus_hospitals, hospital_info, resource_type,
            objective_weights={
                'minimize_shortage': 1.0,   # High: address shortages
                'minimize_cost': 1.0,       # High: but keep costs reasonable
                'maximize_coverage': 1.0,   # High: help hospitals
                'fairness': 1.0             # High: balanced approach
            },
            strategy_name='Balanced'
        )
        if strategy3['status'] == 'optimal':
            strategy3['strategy_name'] = 'Balanced'
            strategy3['strategy_description'] = 'Balanced approach considering cost, coverage, and urgency'
            strategies.append(strategy3)
        
        # Calculate performance-based scores for each strategy
        if strategies:
            # Find min/max for normalization
            costs = [s.get('summary', {}).get('total_cost', 0) for s in strategies]
            hospitals_helped = [s.get('summary', {}).get('hospitals_helped', 0) for s in strategies]
            transfers = [s.get('summary', {}).get('total_transfers', 0) for s in strategies]
            
            min_cost, max_cost = min(costs), max(costs) if costs else (0, 1)
            min_hospitals, max_hospitals = min(hospitals_helped), max(hospitals_helped) if hospitals_helped else (0, 1)
            min_transfers, max_transfers = min(transfers), max(transfers) if transfers else (0, 1)
            
            for strategy in strategies:
                summary = strategy.get('summary', {})
                cost = summary.get('total_cost', 0)
                hospitals = summary.get('hospitals_helped', 0)
                num_transfers = summary.get('total_transfers', 0)
                
                # Normalize scores (0-1 scale)
                if max_cost > min_cost:
                    cost_score = 1.0 - ((cost - min_cost) / (max_cost - min_cost))  # Lower cost = higher score
                else:
                    cost_score = 1.0
                
                if max_hospitals > min_hospitals:
                    coverage_score = (hospitals - min_hospitals) / (max_hospitals - min_hospitals)  # More hospitals = higher score
                else:
                    coverage_score = 1.0
                
                if max_transfers > min_transfers:
                    speed_score = 1.0 - ((num_transfers - min_transfers) / (max_transfers - min_transfers))  # Fewer transfers = faster
                else:
                    speed_score = 1.0
                
                strategy['cost_score'] = round(cost_score, 2)
                strategy['coverage_score'] = round(coverage_score, 2)
                strategy['speed_score'] = round(speed_score, 2)
                
                # Overall score (weighted average)
                strategy['overall_score'] = round(
                    cost_score * 0.3 +
                    speed_score * 0.3 +
                    coverage_score * 0.4,
                    2
                )
        
        # Sort by overall score
        strategies.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        
        # Enhanced validation: Check if strategies are identical and log detailed differences
        if len(strategies) > 1:
            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    s1 = strategies[i]
                    s2 = strategies[j]
                    s1_summary = s1.get('summary', {})
                    s2_summary = s2.get('summary', {})
                    
                    # Check if allocations are identical
                    s1_allocs = sorted([(a['from_hospital_id'], a['to_hospital_id'], a['quantity']) 
                                       for a in s1.get('allocations', [])])
                    s2_allocs = sorted([(a['from_hospital_id'], a['to_hospital_id'], a['quantity']) 
                                       for a in s2.get('allocations', [])])
                    
                    if s1_allocs == s2_allocs:
                        print(f"[WARNING] Strategy '{s1.get('strategy_name')}' and '{s2.get('strategy_name')}' have IDENTICAL allocations!")
                        print(f"  Cost: ${s1_summary.get('total_cost')} vs ${s2_summary.get('total_cost')}")
                        print(f"  Hospitals helped: {s1_summary.get('hospitals_helped')} vs {s2_summary.get('hospitals_helped')}")
                        print(f"  Shortage reduction: {s1_summary.get('shortage_reduction_percent')}% vs {s2_summary.get('shortage_reduction_percent')}%")
                        print(f"  Transfers: {s1_summary.get('total_transfers')} vs {s2_summary.get('total_transfers')}")
                    else:
                        # Log differences for debugging
                        print(f"[INFO] Strategy '{s1.get('strategy_name')}' vs '{s2.get('strategy_name')}':")
                        print(f"  Cost difference: ${abs(s1_summary.get('total_cost', 0) - s2_summary.get('total_cost', 0)):.2f}")
                        print(f"  Hospitals difference: {abs(s1_summary.get('hospitals_helped', 0) - s2_summary.get('hospitals_helped', 0))}")
                        print(f"  Transfer count difference: {abs(s1_summary.get('total_transfers', 0) - s2_summary.get('total_transfers', 0))}")
        
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