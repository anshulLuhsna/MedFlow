"""
Test script to verify data loader pagination and data retrieval
"""

import sys
import os
from pathlib import Path

# Add ml_core directory to path
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

from utils.data_loader import DataLoader


def test_basic_data_loading():
    """Test basic data loading with pagination"""
    print("=" * 70)
    print("Test 1: Basic Data Loading (Pagination)")
    print("=" * 70)
    
    loader = DataLoader()
    
    # Test inventory history
    print("\n1. Testing inventory_history...")
    inventory = loader.get_inventory_history(verbose=True)
    
    print(f"\n   Results:")
    print(f"   - Total records: {len(inventory):,}")
    print(f"   - Unique hospitals: {inventory['hospital_id'].nunique()}")
    print(f"   - Unique resource types: {inventory['resource_type_id'].nunique()}")
    print(f"   - Date range: {inventory['date'].min()} to {inventory['date'].max()}")
    
    # Expected: ~182,000 records, 100 hospitals, 5 resource types
    assert len(inventory) >= 180000, f"Expected >= 180,000 records, got {len(inventory)}"
    assert inventory['hospital_id'].nunique() == 100, f"Expected 100 hospitals, got {inventory['hospital_id'].nunique()}"
    assert inventory['resource_type_id'].nunique() == 5, f"Expected 5 resource types, got {inventory['resource_type_id'].nunique()}"
    print("   ✓ PASSED")
    
    # Test admissions history
    print("\n2. Testing patient_admissions...")
    admissions = loader.get_admissions_history(verbose=True)
    
    print(f"\n   Results:")
    print(f"   - Total records: {len(admissions):,}")
    print(f"   - Unique hospitals: {admissions['hospital_id'].nunique()}")
    print(f"   - Date range: {admissions['admission_date'].min()} to {admissions['admission_date'].max()}")
    
    # Expected: ~36,400 records, 100 hospitals
    assert len(admissions) >= 36000, f"Expected >= 36,000 records, got {len(admissions)}"
    assert admissions['hospital_id'].nunique() == 100, f"Expected 100 hospitals, got {admissions['hospital_id'].nunique()}"
    print("   ✓ PASSED")
    
    return inventory, admissions


def test_resource_type_filtering(inventory):
    """Test filtering by resource type"""
    print("\n" + "=" * 70)
    print("Test 2: Resource Type Filtering")
    print("=" * 70)
    
    loader = DataLoader()
    resource_types = loader.get_resource_types()
    
    print("\nResource types in database:")
    for _, rt in resource_types.iterrows():
        print(f"  - {rt['name']} (ID: {rt['id']})")
    
    print("\nTesting each resource type:")
    resource_counts = {}
    
    for _, rt in resource_types.iterrows():
        rt_name = rt['name']
        rt_id = rt['id']
        
        filtered = inventory[inventory['resource_type_id'] == rt_id]
        unique_hospitals = filtered['hospital_id'].nunique()
        record_count = len(filtered)
        
        resource_counts[rt_name] = {
            'id': rt_id,
            'records': record_count,
            'hospitals': unique_hospitals
        }
        
        print(f"\n  {rt_name.upper()} (ID: {rt_id}):")
        print(f"    - Records: {record_count:,}")
        print(f"    - Unique hospitals: {unique_hospitals}")
        
        # Expected: ~36,400 records per resource type, 100 hospitals
        assert record_count >= 36000, f"Expected >= 36,000 records for {rt_name}, got {record_count}"
        assert unique_hospitals == 100, f"Expected 100 hospitals for {rt_name}, got {unique_hospitals}"
        print(f"    ✓ PASSED")
    
    return resource_counts


def test_prepare_training_data():
    """Test prepare_training_data for each resource type"""
    print("\n" + "=" * 70)
    print("Test 3: Training Data Preparation")
    print("=" * 70)
    
    loader = DataLoader()
    resource_types = loader.get_resource_types()
    
    results = {}
    
    for _, rt in resource_types.iterrows():
        rt_name = rt['name']
        print(f"\n{'='*70}")
        print(f"Testing: {rt_name.upper()}")
        print("=" * 70)
        
        try:
            X, y, metadata = loader.prepare_training_data(rt_name, verbose=True)
            
            results[rt_name] = {
                'status': 'success',
                'sequences': len(X),
                'features': X.shape[2] if len(X.shape) > 2 else X.shape[1],
                'hospitals': metadata['hospital_id'].nunique(),
                'X_shape': X.shape,
                'y_shape': y.shape
            }
            
            print(f"\n✓ SUCCESS for {rt_name}:")
            print(f"  - Training sequences: {len(X):,}")
            print(f"  - Features per timestep: {X.shape[2] if len(X.shape) > 2 else X.shape[1]}")
            print(f"  - Hospitals: {metadata['hospital_id'].nunique()}")
            print(f"  - X shape: {X.shape}")
            print(f"  - y shape: {y.shape}")
            
            # Validate minimum requirements
            assert len(X) >= 30000, f"Expected >= 30,000 sequences for {rt_name}, got {len(X)}"
            assert metadata['hospital_id'].nunique() == 100, f"Expected 100 hospitals for {rt_name}, got {metadata['hospital_id'].nunique()}"
            
        except Exception as e:
            results[rt_name] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"\n✗ FAILED for {rt_name}: {e}")
    
    return results


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("DATA LOADER TEST SUITE")
    print("=" * 70)
    print("\nThis test verifies:")
    print("  1. Pagination retrieves all records (182K inventory, 36K admissions)")
    print("  2. All 5 resource types are accessible")
    print("  3. Training data preparation works for all resource types")
    print("  4. Expected sequence counts (~32K sequences from 100 hospitals)")
    
    try:
        # Test 1: Basic data loading
        inventory, admissions = test_basic_data_loading()
        
        # Test 2: Resource type filtering
        resource_counts = test_resource_type_filtering(inventory)
        
        # Test 3: Training data preparation
        training_results = test_prepare_training_data()
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        print("\n✓ All basic tests passed!")
        print(f"  - Inventory records: {len(inventory):,}")
        print(f"  - Admissions records: {len(admissions):,}")
        print(f"  - Resource types tested: {len(resource_counts)}")
        
        print("\nTraining Data Preparation Results:")
        success_count = sum(1 for r in training_results.values() if r['status'] == 'success')
        failed_count = len(training_results) - success_count
        
        for rt_name, result in training_results.items():
            if result['status'] == 'success':
                print(f"  ✓ {rt_name}: {result['sequences']:,} sequences, {result['hospitals']} hospitals")
            else:
                print(f"  ✗ {rt_name}: {result.get('error', 'Unknown error')}")
        
        print(f"\nTotal: {success_count} succeeded, {failed_count} failed")
        
        if failed_count == 0:
            print("\n🎉 ALL TESTS PASSED! Data loader is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {failed_count} resource type(s) failed. Check errors above.")
            return 1
            
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

