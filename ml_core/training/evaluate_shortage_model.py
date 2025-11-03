"""
Evaluation script for shortage detection model
Loads saved model and evaluates it on test data
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add ml_core directory to Python path for absolute imports
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

# Add parent directory for when running as module
parent_dir = ml_core_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from models.shortage_detector import ShortageDetector
from utils.data_loader import DataLoader
from utils.feature_engineering import engineer_shortage_features
from config import RESOURCE_TYPES, MODELS_DIR
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def evaluate_shortage_model(save_results: bool = True):
    """
    Evaluate the trained shortage detection model
    
    Args:
        save_results: Whether to save results to JSON file
    
    Returns:
        Dict with evaluation metrics and status
    """
    print("\n" + "="*60)
    print("EVALUATING SHORTAGE DETECTION MODEL")
    print("="*60 + "\n")
    
    model_path = MODELS_DIR / "shortage_detector"
    
    # Check if model exists
    if not (model_path / "model.pkl").exists():
        print(f"✗ Model not found at {model_path / 'model.pkl'}")
        return {
            'status': 'error',
            'error': 'Model not found'
        }
    
    try:
        # Initialize
        data_loader = DataLoader()
        detector = ShortageDetector()
        
        # Load trained model
        print("Loading trained model...")
        detector.load()
        print("✓ Model loaded successfully\n")
        
        # Prepare evaluation data
        print("Preparing evaluation data...")
        
        # Get hospitals
        hospitals = data_loader.get_hospitals()
        print(f"  Loaded {len(hospitals)} hospitals")
        
        # Get latest inventory data
        inventory_data = []
        for resource_type in RESOURCE_TYPES:
            print(f"  Loading inventory history for {resource_type}...")
            inv = data_loader.get_inventory_history(
                resource_type=resource_type,
                verbose=False
            )
            if not inv.empty:
                # Get latest record for each hospital
                latest = inv.sort_values('date').groupby('hospital_id').last().reset_index()
                inventory_data.append(latest)
        
        if not inventory_data:
            print("✗ No inventory data found")
            return {
                'status': 'error',
                'error': 'No inventory data available'
            }
        
        inventory_df = pd.concat(inventory_data, ignore_index=True)
        print(f"  Loaded {len(inventory_df)} inventory records")
        
        # Engineer features
        print("  Engineering features...")
        features = engineer_shortage_features(
            inventory_df,
            resource_types=RESOURCE_TYPES,
            hospital_info=hospitals
        )
        
        if features.empty:
            print("✗ No features could be engineered")
            return {
                'status': 'error',
                'error': 'Feature engineering failed'
            }
        
        print(f"  Generated {len(features)} feature samples\n")
        
        # Create labels
        labels = detector.create_risk_labels(features)
        
        # Split data (same split as training for consistency)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Test set: {len(X_test)} samples\n")
        
        # Evaluate model
        print("Evaluating model on test set...")
        metrics = detector.evaluate(X_test, y_test)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        
        print(f"\nPer-Class Performance:")
        report = metrics['classification_report']
        risk_levels = ['low', 'medium', 'high', 'critical']
        
        for risk_level in risk_levels:
            if risk_level in report:
                class_metrics = report[risk_level]
                print(f"\n  {risk_level.capitalize()}:")
                print(f"    Precision: {class_metrics['precision']:.4f}")
                print(f"    Recall: {class_metrics['recall']:.4f}")
                print(f"    F1-Score: {class_metrics['f1-score']:.4f}")
                print(f"    Support: {class_metrics['support']}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"  Shape: {cm.shape}")
        print(f"  Matrix:\n{cm}")
        
        # Class distribution
        print(f"\nTest Set Class Distribution:")
        unique, counts = np.unique(y_test, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} ({count/len(y_test):.2%})")
        
        # Save results
        if save_results:
            results_dir = ml_core_dir / "evaluation_results"
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f"shortage_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'test_set_size': len(X_test),
                'class_distribution': {
                    label: int(count) for label, count in zip(unique, counts)
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to {results_file}")
        
        print("="*60)
        
        return {
            'status': 'success',
            'metrics': metrics
        }
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }
    except Exception as e:
        print(f"✗ Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate shortage detection model')
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save evaluation results to files'
    )
    
    args = parser.parse_args()
    
    evaluate_shortage_model(save_results=not args.no_save)




