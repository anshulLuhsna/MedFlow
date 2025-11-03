"""
Training script for shortage detection model
"""

import sys
import os
from pathlib import Path

# Add ml_core directory to Python path for absolute imports
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

# Add parent directory for when running as module
parent_dir = ml_core_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from models.shortage_detector import ShortageDetector
from models.demand_forecaster import DemandForecaster
from utils.data_loader import DataLoader
from utils.shortage_features import engineer_shortage_features
from config import RESOURCE_TYPES
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def prepare_shortage_training_data():
    """Prepare training data for shortage detection"""

    print("Preparing shortage detection training data...")

    data_loader = DataLoader()

    # Get current inventory
    print("Loading inventory data...")
    inventory = data_loader.get_current_inventory()

    # Get admissions history
    print("Loading admissions history...")
    admissions = data_loader.get_admissions_history()

    # Get hospital info
    print("Loading hospital info...")
    hospitals = data_loader.get_hospitals()

    # Create demand predictions using historical consumption
    print("Creating demand predictions from historical data...")
    inventory_history = data_loader.get_inventory_history()
    
    # Get resource type mapping (ID -> name)
    resource_types_df = data_loader.get_resource_types()
    resource_type_map = dict(zip(resource_types_df['id'], resource_types_df['name']))

    # Create 14-day predictions for each hospital-resource pair
    demand_predictions_list = []

    for (hospital_id, resource_type_id), group in inventory_history.groupby(['hospital_id', 'resource_type_id']):
        # Map resource_type_id to resource_type name
        resource_type_name = resource_type_map.get(resource_type_id, str(resource_type_id))
        
        avg_consumption = group['consumption'].mean()

        # Create 14-day forecast (simple average)
        for day in range(14):
            demand_predictions_list.append({
                'hospital_id': hospital_id,
                'resource_type': resource_type_name,
                'day': day,
                'predicted_consumption': avg_consumption + np.random.normal(0, avg_consumption * 0.1)
            })

    demand_predictions = pd.DataFrame(demand_predictions_list)

    print(f"Created {len(demand_predictions)} demand predictions")

    # Engineer features
    print("Engineering features...")
    features = engineer_shortage_features(
        current_inventory=inventory,
        demand_predictions=demand_predictions,
        admissions_history=admissions,
        hospital_info=hospitals
    )

    return features


def train_shortage_detector():
    """Train shortage detection model"""
    
    print(f"\n{'='*60}")
    print("Training Shortage Detector")
    print(f"{'='*60}\n")
    
    # Prepare data
    features = prepare_shortage_training_data()
    
    print(f"Total samples: {len(features)}")
    
    # Initialize detector
    detector = ShortageDetector()
    
    # Create labels from features
    labels = detector.create_risk_labels(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Train
    print("\nTraining model...")
    metrics = detector.train(X_train, y_train, verbose=1)
    
    # Evaluate
    print("\nEvaluating model...")
    test_metrics = detector.evaluate(X_test, y_test)
    
    print("\nTest Set Metrics:")
    print(f"  Overall Accuracy: {test_metrics['overall_accuracy']:.4f}")
    print(f"  Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
    
    print("\nPer-Class Performance:")
    for risk_level in ['low', 'medium', 'high', 'critical']:
        if risk_level in test_metrics['classification_report']:
            report = test_metrics['classification_report'][risk_level]
            print(f"  {risk_level.capitalize()}:")
            print(f"    Precision: {report['precision']:.3f}")
            print(f"    Recall: {report['recall']:.3f}")
            print(f"    F1-Score: {report['f1-score']:.3f}")
    
    return detector, test_metrics


if __name__ == "__main__":
    train_shortage_detector()