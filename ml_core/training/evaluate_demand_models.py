"""
Evaluation script for demand forecasting models
Loads saved models and evaluates them on test data
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Force CPU usage to avoid CUDA/libdevice issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add ml_core directory to Python path for absolute imports
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

# Add parent directory for when running as module
parent_dir = ml_core_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from models.demand_forecaster import DemandForecaster
from utils.data_loader import DataLoader
from config import RESOURCE_TYPES, DEMAND_FORECAST_CONFIG, MODELS_DIR
import numpy as np
from sklearn.model_selection import train_test_split


def evaluate_demand_model(resource_type: str, save_results: bool = True):
    """
    Evaluate a trained demand forecasting model
    
    Args:
        resource_type: Type of resource to evaluate
        save_results: Whether to save results to JSON file
    
    Returns:
        Dict with evaluation metrics and status
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Demand Forecaster: {resource_type.upper()}")
    print(f"{'='*60}\n")
    
    model_path = MODELS_DIR / "demand_forecaster" / resource_type
    
    # Check if model exists
    if not model_path.exists():
        print(f"✗ Model not found at {model_path}")
        return {
            'status': 'error',
            'resource_type': resource_type,
            'error': 'Model not found'
        }
    
    try:
        # Initialize
        data_loader = DataLoader()
        forecaster = DemandForecaster(resource_type)
        
        # Load trained model
        print("Loading trained model...")
        forecaster.load()
        print("✓ Model loaded successfully\n")
        
        # Load and prepare data
        print("Loading test data...")
        X, y, metadata = data_loader.prepare_training_data(
            resource_type=resource_type,
            sequence_length=DEMAND_FORECAST_CONFIG['sequence_length'],
            verbose=False
        )
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Total samples: {len(X)}\n")
        
        # Split data (same split as training for consistency)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"Test set: {len(X_test)} samples\n")
        
        # Evaluate model
        print("Evaluating model on test set...")
        metrics = forecaster.evaluate(X_test, y_test)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Resource Type: {resource_type}")
        print(f"\nMetrics:")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
        print(f"  MAE (Mean Absolute Error): {metrics['mae']:.4f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        
        # Additional statistics
        print(f"\nTest Set Statistics:")
        print(f"  Actual demand - Mean: {np.mean(y_test):.2f}, Std: {np.std(y_test):.2f}")
        print(f"  Actual demand - Min: {np.min(y_test):.2f}, Max: {np.max(y_test):.2f}")
        
        # Predictions statistics
        predictions = forecaster.predict(X_test, return_confidence=False)
        print(f"  Predicted demand - Mean: {np.mean(predictions):.2f}, Std: {np.std(predictions):.2f}")
        print(f"  Predicted demand - Min: {np.min(predictions):.2f}, Max: {np.max(predictions):.2f}")
        
        # Save results
        if save_results:
            results_dir = ml_core_dir / "evaluation_results"
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f"demand_forecast_{resource_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            results = {
                'resource_type': resource_type,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'test_set_size': len(X_test),
                'test_set_stats': {
                    'actual_mean': float(np.mean(y_test)),
                    'actual_std': float(np.std(y_test)),
                    'actual_min': float(np.min(y_test)),
                    'actual_max': float(np.max(y_test)),
                    'predicted_mean': float(np.mean(predictions)),
                    'predicted_std': float(np.std(predictions)),
                    'predicted_min': float(np.min(predictions)),
                    'predicted_max': float(np.max(predictions))
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to {results_file}")
        
        print("="*60)
        
        return {
            'status': 'success',
            'resource_type': resource_type,
            'metrics': metrics
        }
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return {
            'status': 'error',
            'resource_type': resource_type,
            'error': str(e)
        }
    except Exception as e:
        print(f"✗ Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'resource_type': resource_type,
            'error': str(e)
        }


def evaluate_all_demand_models(save_results: bool = True):
    """
    Evaluate all trained demand forecasting models
    
    Args:
        save_results: Whether to save results to JSON files
    """
    print("\n" + "="*60)
    print("EVALUATING ALL DEMAND FORECASTING MODELS")
    print("="*60)
    
    results = {}
    
    for resource_type in RESOURCE_TYPES:
        result = evaluate_demand_model(resource_type, save_results=save_results)
        results[resource_type] = result
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    successful = [r for r in results.values() if r['status'] == 'success']
    failed = [r for r in results.values() if r['status'] == 'error']
    
    print(f"\nSuccessful: {len(successful)}/{len(RESOURCE_TYPES)}")
    print(f"Failed: {len(failed)}/{len(RESOURCE_TYPES)}")
    
    if successful:
        print("\nMetrics Summary:")
        print("-" * 60)
        print(f"{'Resource':<15} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Dir. Acc.':<10}")
        print("-" * 60)
        
        for resource_type, result in results.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                print(f"{resource_type:<15} {metrics['rmse']:>8.2f} {metrics['mae']:>8.2f} "
                      f"{metrics['mape']:>8.2f}% {metrics['directional_accuracy']:>8.2%}")
    
    if failed:
        print("\nFailed Evaluations:")
        for resource_type, result in results.items():
            if result['status'] == 'error':
                print(f"  {resource_type}: {result.get('error', 'Unknown error')}")
    
    print("="*60)
    
    # Save summary
    if save_results:
        results_dir = ml_core_dir / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        summary_file = results_dir / f"demand_forecast_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(RESOURCE_TYPES),
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to {summary_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate demand forecasting models')
    parser.add_argument(
        '--resource',
        type=str,
        choices=RESOURCE_TYPES,
        help='Evaluate a specific resource type (default: all)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save evaluation results to files'
    )
    
    args = parser.parse_args()
    
    if args.resource:
        evaluate_demand_model(args.resource, save_results=not args.no_save)
    else:
        evaluate_all_demand_models(save_results=not args.no_save)

