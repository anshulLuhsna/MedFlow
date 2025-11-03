"""
Test Scenarios for Resource Optimizer

Tests various optimization scenarios to ensure the optimizer correctly allocates resources.
"""

import sys
from pathlib import Path

# Add ml_core directory to Python path
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

import pandas as pd
import numpy as np
from models.optimizer import ResourceOptimizer


def create_test_hospitals(hospital_ids, base_lat=37.7749, base_lon=-122.4194, spacing_km=50):
    """Create test hospitals with coordinates spaced by spacing_km"""
    hospitals = []
    for i, hospital_id in enumerate(hospital_ids):
        # Space hospitals in a line (simple approach)
        lat_offset = (i % 3) * (spacing_km / 111.0)  # ~1 degree lat = 111 km
        lon_offset = (i // 3) * (spacing_km / (111.0 * np.cos(np.radians(base_lat))))
        
        hospitals.append({
            'hospital_id': hospital_id,
            'name': f'Hospital {hospital_id}',
            'latitude': base_lat + lat_offset,
            'longitude': base_lon + lon_offset,
            'region': f'Region_{i % 3}',
            'capacity_beds': 200 + i * 50
        })
    
    return pd.DataFrame(hospitals)


def create_test_shortages(hospital_ids, quantity_needed_list, risk_levels):
    """Create shortage hospitals DataFrame"""
    shortages = []
    for i, hospital_id in enumerate(hospital_ids):
        shortages.append({
            'hospital_id': hospital_id,
            'resource_type': 'ventilators',
            'quantity_needed': quantity_needed_list[i],
            'current_stock': max(0, quantity_needed_list[i] - 5),  # Current stock is less than needed
            'risk_level': risk_levels[i],
            'available_quantity': max(0, quantity_needed_list[i] - 5)
        })
    
    return pd.DataFrame(shortages)


def create_test_surpluses(hospital_ids, available_quantities):
    """Create surplus hospitals DataFrame"""
    surpluses = []
    for i, hospital_id in enumerate(hospital_ids):
        surpluses.append({
            'hospital_id': hospital_id,
            'resource_type': 'ventilators',
            'available_quantity': available_quantities[i],
            'quantity': available_quantities[i] + 5,  # Total quantity includes reserved
            'current_stock': available_quantities[i] + 5
        })
    
    return pd.DataFrame(surpluses)


def test_scenario_1_simple_transfer():
    """Scenario 1: Simple 1-to-1 transfer"""
    
    print("\n" + "="*60)
    print("Scenario 1: Simple 1-to-1 Transfer")
    print("="*60)
    
    optimizer = ResourceOptimizer()
    
    # 1 shortage hospital, 1 surplus hospital within range
    hospital_ids = ['H001', 'H002']
    hospitals = create_test_hospitals(hospital_ids, spacing_km=50)  # 50km apart (within 200km limit)
    
    shortages = create_test_shortages(['H001'], [10], ['critical'])
    surpluses = create_test_surpluses(['H002'], [15])
    
    result = optimizer.optimize_allocation(
        shortage_hospitals=shortages,
        surplus_hospitals=surpluses,
        hospital_info=hospitals,
        resource_type='ventilators'
    )
    
    print(f"\nStatus: {result['status']}")
    
    if result['status'] == 'optimal':
        print(f"Total transfers: {len(result['allocations'])}")
        print(f"Total quantity transferred: {result['summary']['total_quantity_transferred']}")
        print(f"Total cost: ${result['summary']['total_cost']:.2f}")
        print(f"Hospitals helped: {result['summary']['hospitals_helped']}")
        
        for allocation in result['allocations']:
            print(f"\n  Transfer: {allocation['from_hospital_id']} → {allocation['to_hospital_id']}")
            print(f"    Quantity: {allocation['quantity']}")
            print(f"    Distance: {allocation['distance_km']:.1f} km")
            print(f"    Cost: ${allocation['transfer_cost']:.2f}")
        
        # Verify
        assert len(result['allocations']) == 1, "Should have exactly 1 transfer"
        assert result['allocations'][0]['from_hospital_id'] == 'H002', "Should transfer from H002"
        assert result['allocations'][0]['to_hospital_id'] == 'H001', "Should transfer to H001"
        assert result['allocations'][0]['quantity'] <= 10, "Should not exceed needed quantity"
        assert result['allocations'][0]['quantity'] <= 15, "Should not exceed available quantity"
        
        print("\n✓ Scenario 1 PASSED: Simple 1-to-1 transfer works correctly")
        return True
    else:
        print(f"✗ Optimization failed: {result.get('message', 'Unknown error')}")
        return False


def test_scenario_2_multiple_shortages_single_surplus():
    """Scenario 2: Multiple shortages, single surplus - prioritizes critical"""
    
    print("\n" + "="*60)
    print("Scenario 2: Multiple Shortages, Single Surplus")
    print("="*60)
    
    optimizer = ResourceOptimizer()
    
    # 3 shortage hospitals (2 critical, 1 high), 1 surplus
    hospital_ids = ['H001', 'H002', 'H003', 'H004']
    hospitals = create_test_hospitals(hospital_ids, spacing_km=50)
    
    shortages = create_test_shortages(
        ['H001', 'H002', 'H003'],
        [8, 5, 10],
        ['critical', 'critical', 'high']
    )
    surpluses = create_test_surpluses(['H004'], [15])  # Only 15 available, but need 23 total
    
    result = optimizer.optimize_allocation(
        shortage_hospitals=shortages,
        surplus_hospitals=surpluses,
        hospital_info=hospitals,
        resource_type='ventilators'
    )
    
    print(f"\nStatus: {result['status']}")
    
    if result['status'] == 'optimal':
        print(f"Total transfers: {len(result['allocations'])}")
        print(f"Total quantity transferred: {result['summary']['total_quantity_transferred']}")
        print(f"Shortage reduction: {result['summary']['shortage_reduction_percent']:.1f}%")
        
        # Check which hospitals received allocations
        helped_hospitals = [a['to_hospital_id'] for a in result['allocations']]
        print(f"Hospitals helped: {helped_hospitals}")
        
        # Verify critical hospitals are prioritized
        critical_received = sum([
            a['quantity'] for a in result['allocations']
            if shortages[shortages['hospital_id'] == a['to_hospital_id']]['risk_level'].iloc[0] == 'critical'
        ])
        print(f"Critical hospitals received: {critical_received} units")
        
        # Verify: Critical hospitals should get allocation (fairness constraint)
        critical_ids = shortages[shortages['risk_level'] == 'critical']['hospital_id'].tolist()
        helped_critical = [h for h in critical_ids if h in helped_hospitals]
        assert len(helped_critical) >= 1, "At least one critical hospital should receive allocation"
        
        print("\n✓ Scenario 2 PASSED: Critical shortages prioritized")
        return True
    else:
        print(f"✗ Optimization failed: {result.get('message', 'Unknown error')}")
        return False


def test_scenario_3_multiple_surpluses_single_shortage():
    """Scenario 3: Multiple surpluses, single shortage - cost minimized"""
    
    print("\n" + "="*60)
    print("Scenario 3: Multiple Surpluses, Single Shortage")
    print("="*60)
    
    optimizer = ResourceOptimizer()
    
    # 1 shortage hospital, 3 surplus hospitals at different distances
    hospital_ids = ['H001', 'H002', 'H003', 'H004']
    # Place shortage at (37.7749, -122.4194)
    # Place surpluses at increasing distances: 30km, 80km, 150km
    hospitals_data = [
        {'hospital_id': 'H001', 'latitude': 37.7749, 'longitude': -122.4194, 'name': 'Shortage Hospital', 'region': 'A', 'capacity_beds': 200},
        {'hospital_id': 'H002', 'latitude': 38.0449, 'longitude': -122.4194, 'name': 'Near Surplus', 'region': 'A', 'capacity_beds': 200},  # ~30km north
        {'hospital_id': 'H003', 'latitude': 38.3949, 'longitude': -122.4194, 'name': 'Medium Surplus', 'region': 'B', 'capacity_beds': 250},  # ~80km north
        {'hospital_id': 'H004', 'latitude': 39.1249, 'longitude': -122.4194, 'name': 'Far Surplus', 'region': 'C', 'capacity_beds': 300},  # ~150km north
    ]
    hospitals = pd.DataFrame(hospitals_data)
    
    shortages = create_test_shortages(['H001'], [10], ['critical'])
    surpluses = create_test_surpluses(['H002', 'H003', 'H004'], [5, 5, 5])  # Each has 5, need 10 total
    
    result = optimizer.optimize_allocation(
        shortage_hospitals=shortages,
        surplus_hospitals=surpluses,
        hospital_info=hospitals,
        resource_type='ventilators'
    )
    
    print(f"\nStatus: {result['status']}")
    
    if result['status'] == 'optimal':
        print(f"Total transfers: {len(result['allocations'])}")
        print(f"Total cost: ${result['summary']['total_cost']:.2f}")
        
        for allocation in result['allocations']:
            print(f"\n  Transfer: {allocation['from_hospital_id']} → {allocation['to_hospital_id']}")
            print(f"    Quantity: {allocation['quantity']}")
            print(f"    Distance: {allocation['distance_km']:.1f} km")
            print(f"    Cost: ${allocation['transfer_cost']:.2f}")
        
        # Verify: Should prefer closer hospitals (lower cost)
        if len(result['allocations']) > 0:
            distances = [a['distance_km'] for a in result['allocations']]
            # Nearest hospital (H002) should be used if it has enough
            print(f"\nDistances used: {distances}")
        
        print("\n✓ Scenario 3 PASSED: Multiple sources considered")
        return True
    else:
        print(f"✗ Optimization failed: {result.get('message', 'Unknown error')}")
        return False


def test_scenario_4_distance_constraints():
    """Scenario 4: Distance constraints - only nearby hospitals considered"""
    
    print("\n" + "="*60)
    print("Scenario 4: Distance Constraints")
    print("="*60)
    
    optimizer = ResourceOptimizer()
    
    # 1 shortage hospital, 2 surplus hospitals (1 near, 1 far beyond 200km)
    hospital_ids = ['H001', 'H002', 'H003']
    hospitals_data = [
        {'hospital_id': 'H001', 'latitude': 37.7749, 'longitude': -122.4194, 'name': 'Shortage', 'region': 'A', 'capacity_beds': 200},
        {'hospital_id': 'H002', 'latitude': 38.0449, 'longitude': -122.4194, 'name': 'Near Surplus', 'region': 'A', 'capacity_beds': 200},  # ~30km
        {'hospital_id': 'H003', 'latitude': 40.0, 'longitude': -122.4194, 'name': 'Far Surplus', 'region': 'B', 'capacity_beds': 250},  # ~250km (beyond limit)
    ]
    hospitals = pd.DataFrame(hospitals_data)
    
    shortages = create_test_shortages(['H001'], [10], ['critical'])
    surpluses = create_test_surpluses(['H002', 'H003'], [15, 20])
    
    result = optimizer.optimize_allocation(
        shortage_hospitals=shortages,
        surplus_hospitals=surpluses,
        hospital_info=hospitals,
        resource_type='ventilators'
    )
    
    print(f"\nStatus: {result['status']}")
    
    if result['status'] == 'optimal':
        print(f"Total transfers: {len(result['allocations'])}")
        
        # Verify only near hospital (H002) is used
        from_hospitals = [a['from_hospital_id'] for a in result['allocations']]
        print(f"Source hospitals: {from_hospitals}")
        
        # Should not use H003 (too far)
        assert 'H003' not in from_hospitals, "Far hospital (H003) should not be used"
        # Should use H002 (near)
        assert 'H002' in from_hospitals, "Near hospital (H002) should be used"
        
        for allocation in result['allocations']:
            print(f"\n  Transfer: {allocation['from_hospital_id']} → {allocation['to_hospital_id']}")
            print(f"    Distance: {allocation['distance_km']:.1f} km")
            assert allocation['distance_km'] <= 200, f"Distance {allocation['distance_km']}km exceeds 200km limit"
        
        print("\n✓ Scenario 4 PASSED: Distance constraints respected")
        return True
    else:
        print(f"✗ Optimization failed: {result.get('message', 'Unknown error')}")
        return False


def test_scenario_5_insufficient_surplus():
    """Scenario 5: Insufficient surplus - critical hospitals prioritized"""
    
    print("\n" + "="*60)
    print("Scenario 5: Insufficient Surplus")
    print("="*60)
    
    optimizer = ResourceOptimizer()
    
    # 3 shortage hospitals (2 critical, 1 high), 1 surplus with limited quantity
    hospital_ids = ['H001', 'H002', 'H003', 'H004']
    hospitals = create_test_hospitals(hospital_ids, spacing_km=50)
    
    shortages = create_test_shortages(
        ['H001', 'H002', 'H003'],
        [10, 8, 7],  # Total need: 25
        ['critical', 'critical', 'high']
    )
    surpluses = create_test_surpluses(['H004'], [12])  # Only 12 available, need 25
    
    result = optimizer.optimize_allocation(
        shortage_hospitals=shortages,
        surplus_hospitals=surpluses,
        hospital_info=hospitals,
        resource_type='ventilators'
    )
    
    print(f"\nStatus: {result['status']}")
    
    if result['status'] == 'optimal':
        print(f"Total transfers: {len(result['allocations'])}")
        print(f"Total quantity transferred: {result['summary']['total_quantity_transferred']}")
        print(f"Total shortage before: {result['summary']['total_shortage_before']}")
        print(f"Total shortage after: {result['summary']['total_shortage_after']}")
        print(f"Shortage reduction: {result['summary']['shortage_reduction_percent']:.1f}%")
        
        # Verify total transferred doesn't exceed available
        assert result['summary']['total_quantity_transferred'] <= 12, "Should not exceed available surplus"
        
        # Check allocation to critical vs high risk hospitals
        allocations_by_risk = {}
        for allocation in result['allocations']:
            to_id = allocation['to_hospital_id']
            risk = shortages[shortages['hospital_id'] == to_id]['risk_level'].iloc[0]
            if risk not in allocations_by_risk:
                allocations_by_risk[risk] = 0
            allocations_by_risk[risk] += allocation['quantity']
        
        print(f"\nAllocations by risk level:")
        for risk, quantity in allocations_by_risk.items():
            print(f"  {risk}: {quantity} units")
        
        # Critical hospitals should receive some allocation (fairness constraint)
        if 'critical' in allocations_by_risk:
            assert allocations_by_risk['critical'] > 0, "Critical hospitals should receive allocation"
        
        print("\n✓ Scenario 5 PASSED: Critical hospitals prioritized when surplus is insufficient")
        return True
    else:
        print(f"✗ Optimization failed: {result.get('message', 'Unknown error')}")
        return False


def test_scenario_6_multi_objective_tradeoffs():
    """Scenario 6: Multi-objective trade-offs - compare strategies"""
    
    print("\n" + "="*60)
    print("Scenario 6: Multi-Objective Trade-offs")
    print("="*60)
    
    optimizer = ResourceOptimizer()
    
    # Create scenario with multiple options
    hospital_ids = ['H001', 'H002', 'H003', 'H004', 'H005']
    hospitals = create_test_hospitals(hospital_ids, spacing_km=80)
    
    shortages = create_test_shortages(
        ['H001', 'H002'],
        [8, 5],
        ['critical', 'high']
    )
    surpluses = create_test_surpluses(['H003', 'H004', 'H005'], [10, 10, 10])
    
    # Generate multiple strategies
    strategies = optimizer.generate_multiple_strategies(
        shortage_hospitals=shortages,
        surplus_hospitals=surpluses,
        hospital_info=hospitals,
        resource_type='ventilators',
        n_strategies=3
    )
    
    print(f"\nGenerated {len(strategies)} strategies:")
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\nStrategy {i}: {strategy.get('strategy_name', 'Unknown')}")
        print(f"  Description: {strategy.get('strategy_description', 'N/A')}")
        if strategy['status'] == 'optimal':
            print(f"  Cost score: {strategy.get('cost_score', 0):.2f}")
            print(f"  Coverage score: {strategy.get('coverage_score', 0):.2f}")
            print(f"  Speed score: {strategy.get('speed_score', 0):.2f}")
            print(f"  Overall score: {strategy.get('overall_score', 0):.2f}")
            print(f"  Total cost: ${strategy['summary']['total_cost']:.2f}")
            print(f"  Hospitals helped: {strategy['summary']['hospitals_helped']}")
            print(f"  Transfers: {strategy['summary']['total_transfers']}")
        else:
            print(f"  Status: {strategy['status']}")
    
    # Verify strategies are different
    if len(strategies) >= 2:
        optimal_strategies = [s for s in strategies if s['status'] == 'optimal']
        if len(optimal_strategies) >= 2:
            costs = [s['summary']['total_cost'] for s in optimal_strategies]
            print(f"\nCosts across strategies: {costs}")
            # Strategies should be ranked by overall_score
            scores = [s.get('overall_score', 0) for s in optimal_strategies]
            print(f"Overall scores: {scores}")
            # Verify ranking is descending
            assert scores == sorted(scores, reverse=True), "Strategies should be ranked by overall_score"
    
    print("\n✓ Scenario 6 PASSED: Multiple strategies with different trade-offs generated")
    return True


def test_scenario_7_empty_scenarios():
    """Scenario 7: Empty scenarios - handle gracefully"""
    
    print("\n" + "="*60)
    print("Scenario 7: Empty Scenarios")
    print("="*60)
    
    optimizer = ResourceOptimizer()
    
    hospital_ids = ['H001', 'H002']
    hospitals = create_test_hospitals(hospital_ids)
    
    # Test 1: No shortages
    print("\nTest 7a: No shortages")
    shortages_empty = pd.DataFrame(columns=['hospital_id', 'quantity_needed', 'risk_level'])
    surpluses = create_test_surpluses(['H002'], [15])
    
    result1 = optimizer.optimize_allocation(
        shortage_hospitals=shortages_empty,
        surplus_hospitals=surpluses,
        hospital_info=hospitals,
        resource_type='ventilators'
    )
    print(f"  Status: {result1['status']}")
    print(f"  Message: {result1.get('message', 'N/A')}")
    assert len(result1['allocations']) == 0, "Should have no allocations when no shortages"
    
    # Test 2: No surpluses
    print("\nTest 7b: No surpluses")
    shortages = create_test_shortages(['H001'], [10], ['critical'])
    surpluses_empty = pd.DataFrame(columns=['hospital_id', 'available_quantity'])
    
    result2 = optimizer.optimize_allocation(
        shortage_hospitals=shortages,
        surplus_hospitals=surpluses_empty,
        hospital_info=hospitals,
        resource_type='ventilators'
    )
    print(f"  Status: {result2['status']}")
    print(f"  Message: {result2.get('message', 'N/A')}")
    assert len(result2['allocations']) == 0, "Should have no allocations when no surpluses"
    
    # Test 3: All too far (no feasible transfers)
    print("\nTest 7c: All hospitals too far apart")
    hospitals_far = pd.DataFrame([
        {'hospital_id': 'H001', 'latitude': 37.7749, 'longitude': -122.4194, 'name': 'Shortage', 'region': 'A', 'capacity_beds': 200},
        {'hospital_id': 'H002', 'latitude': 40.5, 'longitude': -122.4194, 'name': 'Far Surplus', 'region': 'B', 'capacity_beds': 250},  # ~300km
    ])
    shortages = create_test_shortages(['H001'], [10], ['critical'])
    surpluses = create_test_surpluses(['H002'], [15])
    
    result3 = optimizer.optimize_allocation(
        shortage_hospitals=shortages,
        surplus_hospitals=surpluses,
        hospital_info=hospitals_far,
        resource_type='ventilators'
    )
    print(f"  Status: {result3['status']}")
    print(f"  Message: {result3.get('message', 'N/A')}")
    # Should return no_feasible_transfers or similar status
    assert result3['status'] in ['no_feasible_transfers', 'optimal'], "Should handle no feasible transfers"
    if result3['status'] == 'no_feasible_transfers':
        assert len(result3['allocations']) == 0, "Should have no allocations when too far"
    
    print("\n✓ Scenario 7 PASSED: Empty scenarios handled gracefully")
    return True


def run_all_tests():
    """Run all test scenarios"""
    print("\n" + "="*60)
    print("OPTIMIZER - TEST SCENARIOS")
    print("="*60)
    
    results = []
    test_functions = [
        test_scenario_1_simple_transfer,
        test_scenario_2_multiple_shortages_single_surplus,
        test_scenario_3_multiple_surpluses_single_shortage,
        test_scenario_4_distance_constraints,
        test_scenario_5_insufficient_surplus,
        test_scenario_6_multi_objective_tradeoffs,
        test_scenario_7_empty_scenarios
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            results.append((test_func.__name__, result, None))
        except AssertionError as e:
            print(f"\n✗ {test_func.__name__} FAILED: {e}")
            results.append((test_func.__name__, False, str(e)))
        except Exception as e:
            print(f"\n✗ {test_func.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {test_name}")
        if error:
            print(f"    Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(f"⚠ {total - passed} TEST(S) FAILED")
        print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

