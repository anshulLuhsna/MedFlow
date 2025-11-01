"""
Training script for demand forecasting models
"""

import sys
import os
from pathlib import Path

# Force CPU usage to avoid CUDA/libdevice issues
# Set before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

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
from config import RESOURCE_TYPES, DEMAND_FORECAST_CONFIG
import numpy as np
from sklearn.model_selection import train_test_split


def train_demand_forecaster(resource_type: str):
    """Train demand forecaster for a specific resource type"""
    
    print(f"\n{'='*60}")
    print(f"Training Demand Forecaster: {resource_type.upper()}")
    print(f"{'='*60}\n")
    
    # Initialize
    data_loader = DataLoader()
    forecaster = DemandForecaster(resource_type)
    
    # Load data
    print("Loading training data...")
    X, y, metadata = data_loader.prepare_training_data(
        resource_type=resource_type,
        sequence_length=DEMAND_FORECAST_CONFIG['sequence_length']
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Samples: {len(X)}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Train
    print("\nTraining model...")
    history = forecaster.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = forecaster.evaluate(X_test, y_test)
    
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        # Skip non-numeric metrics when printing
        if metric != 'resource_type' and isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        elif metric == 'resource_type':
            print(f"  {metric}: {value}")
        else:
            print(f"  {metric}: {value}")
    
    return forecaster, metrics


def train_all_forecasters():
    """Train forecasters for all resource types"""
    
    results = {}
    
    for resource_type in RESOURCE_TYPES:
        try:
            forecaster, metrics = train_demand_forecaster(resource_type)
            results[resource_type] = {
                'status': 'success',
                'metrics': metrics
            }
        except Exception as e:
            print(f"\n✗ Error training {resource_type}: {e}")
            results[resource_type] = {
                'status': 'failed',
                'error': str(e)
            }
    
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}\n")
    
    for resource_type, result in results.items():
        status = result['status']
        print(f"{resource_type}: {status}")
        if status == 'success':
            metrics = result['metrics']
            # Safely format metrics, handling potential string values
            try:
                rmse = float(metrics.get('rmse', 0))
                mae = float(metrics.get('mae', 0))
                mape = float(metrics.get('mape', 0))
                dir_acc = float(metrics.get('directional_accuracy', 0))
                print(f"  RMSE: {rmse:.2f}")
                print(f"  MAE: {mae:.2f}")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  Directional Accuracy: {dir_acc:.2%}")
            except (ValueError, TypeError, KeyError) as e:
                print(f"  Error formatting metrics: {e}")
                print(f"  Metrics dict: {metrics}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train demand forecasting models')
    parser.add_argument(
        '--resource',
        type=str,
        choices=RESOURCE_TYPES + ['all'],
        default='all',
        help='Resource type to train (or "all")'
    )
    
    args = parser.parse_args()
    
    if args.resource == 'all':
        train_all_forecasters()
    else:
        train_demand_forecaster(args.resource)