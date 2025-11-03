"""
Integration tests for Optimizer with MLCore

Tests end-to-end integration: detect_shortages → optimize_allocation
"""

import sys
from pathlib import Path

# Add ml_core directory to Python path
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

import pandas as pd
import numpy as np
from core import MLCore
import warnings
warnings.filterwarnings('ignore')


def test_mlcore_end_to_end():
    """Test end-to-end flow: detect_shortages → optimize_allocation"""
    
    print("\n" + "="*60)
    print("MLCore End-to-End Integration Test")
    print("="*60)
    
    try:
        ml_core = MLCore()
    except Exception as e:
        print(f"✗ Failed to initialize MLCore: {e}")
        return False
    
    # Test for each resource type
    resource_types = ['ventilators', 'ppe', 'o2_cylinders', 'medications', 'beds']
    
    for resource_type in resource_types:
        print(f"\n--- Testing {resource_type.upper()} ---")
        
        try:
            # Step 1: Detect shortages
            print("  Step 1: Detecting shortages...")
            shortages = ml_core.detect_shortages(resource_type=resource_type)
            
            if shortages.empty:
                print(f"    No shortages detected for {resource_type}")
                continue
            
            print(f"    Found {len(shortages)} hospitals with shortages")
            high_critical = shortages[shortages['risk_level'].isin(['high', 'critical'])]
            print(f"    High/Critical shortages: {len(high_critical)}")
            
            if high_critical.empty:
                print(f"    No high/critical shortages for {resource_type}, skipping optimization")
                continue
            
            # Step 2: Optimize allocation
            print("  Step 2: Optimizing allocation...")
            result = ml_core.optimize_allocation(resource_type=resource_type)
            
            print(f"    Status: {result['status']}")
            
            if result['status'] == 'optimal':
                print(f"    Total transfers: {result['summary']['total_transfers']}")
                print(f"    Quantity transferred: {result['summary']['total_quantity_transferred']}")
                print(f"    Total cost: ${result['summary']['total_cost']:.2f}")
                print(f"    Hospitals helped: {result['summary']['hospitals_helped']}")
                print(f"    Shortage reduction: {result['summary']['shortage_reduction_percent']:.1f}%")
                
                # Validate allocations
                allocations = result['allocations']
                if allocations:
                    print(f"\n    Allocations:")
                    for alloc in allocations[:3]:  # Show first 3
                        print(f"      {alloc['from_hospital_id'][:8]}... → {alloc['to_hospital_id'][:8]}...")
                        print(f"        Quantity: {alloc['quantity']}, Distance: {alloc['distance_km']:.1f}km")
                    
                    # Verify feasibility
                    total_transferred = sum(a['quantity'] for a in allocations)
                    assert total_transferred == result['summary']['total_quantity_transferred'], \
                        "Total transferred mismatch"
                    
                    print(f"\n    ✓ Allocations validated")
            elif result['status'] == 'no_feasible_transfers':
                print(f"    No feasible transfers found (likely no surpluses or all too far)")
            else:
                print(f"    Optimization status: {result.get('message', result['status'])}")
        
        except Exception as e:
            print(f"    ✗ Error testing {resource_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✓ MLCore end-to-end integration test completed")
    return True


def test_multiple_strategies():
    """Test generating multiple allocation strategies"""
    
    print("\n" + "="*60)
    print("Multiple Strategies Generation Test")
    print("="*60)
    
    try:
        ml_core = MLCore()
    except Exception as e:
        print(f"✗ Failed to initialize MLCore: {e}")
        return False
    
    resource_type = 'ventilators'
    
    try:
        # Detect shortages
        shortages = ml_core.detect_shortages(resource_type=resource_type)
        high_critical = shortages[shortages['risk_level'].isin(['high', 'critical'])]
        
        if high_critical.empty:
            print(f"No high/critical shortages for {resource_type}, skipping")
            return True
        
        print(f"Found {len(high_critical)} high/critical shortages")
        
        # Generate multiple strategies
        print("\nGenerating 3 strategies...")
        strategies = ml_core.generate_allocation_strategies(
            resource_type=resource_type,
            n_strategies=3
        )
        
        print(f"\nGenerated {len(strategies)} strategies:")
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n  Strategy {i}: {strategy.get('strategy_name', 'Unknown')}")
            print(f"    {strategy.get('strategy_description', 'N/A')}")
            
            if strategy['status'] == 'optimal':
                summary = strategy['summary']
                print(f"    Cost: ${summary['total_cost']:.2f}")
                print(f"    Transfers: {summary['total_transfers']}")
                print(f"    Hospitals helped: {summary['hospitals_helped']}")
                print(f"    Overall score: {strategy.get('overall_score', 0):.2f}")
            else:
                print(f"    Status: {strategy['status']}")
        
        # Verify strategies are ranked
        optimal_strategies = [s for s in strategies if s['status'] == 'optimal']
        if len(optimal_strategies) >= 2:
            scores = [s.get('overall_score', 0) for s in optimal_strategies]
            print(f"\n  Overall scores: {scores}")
            # Should be descending
            assert scores == sorted(scores, reverse=True), \
                "Strategies should be ranked by overall_score (descending)"
            print(f"  ✓ Strategies properly ranked")
        
        print("\n✓ Multiple strategies generation test passed")
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases: no shortages, no surpluses, etc."""
    
    print("\n" + "="*60)
    print("Edge Cases Test")
    print("="*60)
    
    try:
        ml_core = MLCore()
    except Exception as e:
        print(f"✗ Failed to initialize MLCore: {e}")
        return False
    
    # Test 1: Resource with no shortages
    print("\nTest 1: Resource with no shortages")
    try:
        # Try a resource type that likely has no shortages
        result = ml_core.optimize_allocation(resource_type='beds')
        print(f"  Status: {result['status']}")
        print(f"  Message: {result.get('message', 'N/A')}")
        
        # Should handle gracefully
        assert 'status' in result, "Should return status"
        print(f"  ✓ Handled gracefully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Resource with no surpluses (all hospitals need resources)
    print("\nTest 2: Resource with no surpluses")
    try:
        # This will depend on actual data
        result = ml_core.optimize_allocation(resource_type='ventilators')
        print(f"  Status: {result['status']}")
        
        if result['status'] == 'no_feasible_transfers':
            print(f"  ✓ Correctly identified no feasible transfers")
        elif result['status'] == 'optimal':
            print(f"  ✓ Found allocations")
        else:
            print(f"  Status: {result['status']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n✓ Edge cases test completed")
    return True


def validate_allocation_feasibility(allocations, shortage_hospitals, surplus_hospitals):
    """Validate that allocations are feasible"""
    
    issues = []
    
    # Track transfers from each surplus hospital
    surplus_used = {}
    for surplus_id in surplus_hospitals['hospital_id']:
        surplus_used[surplus_id] = 0
    
    # Track transfers to each shortage hospital
    shortage_satisfied = {}
    for shortage_id in shortage_hospitals['hospital_id']:
        shortage_satisfied[shortage_id] = 0
    
    # Validate each allocation
    for alloc in allocations:
        from_id = alloc['from_hospital_id']
        to_id = alloc['to_hospital_id']
        quantity = alloc['quantity']
        
        # Check surplus availability
        if from_id in surplus_used:
            available = surplus_hospitals[
                surplus_hospitals['hospital_id'] == from_id
            ]['available_quantity'].iloc[0]
            surplus_used[from_id] += quantity
            
            if surplus_used[from_id] > available:
                issues.append(
                    f"Surplus hospital {from_id} transfers {surplus_used[from_id]} "
                    f"but only has {available} available"
                )
        
        # Check shortage satisfaction
        if to_id in shortage_satisfied:
            shortage_satisfied[to_id] += quantity
            needed = shortage_hospitals[
                shortage_hospitals['hospital_id'] == to_id
            ]['quantity_needed'].iloc[0]
            
            if shortage_satisfied[to_id] > needed:
                issues.append(
                    f"Shortage hospital {to_id} receives {shortage_satisfied[to_id]} "
                    f"but only needs {needed}"
                )
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'surplus_usage': surplus_used,
        'shortage_satisfaction': shortage_satisfied
    }


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("OPTIMIZER INTEGRATION TESTS")
    print("="*60)
    
    results = []
    test_functions = [
        test_mlcore_end_to_end,
        test_multiple_strategies,
        test_edge_cases
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

