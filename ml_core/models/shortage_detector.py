"""
Shortage Detection Model - Random Forest Classifier
Classifies shortage risk level based on current state and predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# Handle both relative and absolute imports
try:
    from ..config import SHORTAGE_DETECTION_CONFIG, MODEL_PATHS, RESOURCE_TYPES
    from ..utils.shortage_features import engineer_shortage_features
except ImportError:
    from config import SHORTAGE_DETECTION_CONFIG, MODEL_PATHS, RESOURCE_TYPES
    from utils.shortage_features import engineer_shortage_features


class ShortageDetector:
    """Random Forest classifier for shortage risk detection"""
    
    def __init__(self):
        self.config = SHORTAGE_DETECTION_CONFIG
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.model_path = MODEL_PATHS["shortage_detector"]
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Fit label encoder
        self.label_encoder.fit(self.config['risk_levels'])
    
    def build_model(self) -> RandomForestClassifier:
        """Build Random Forest classifier"""
        model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            random_state=self.config['random_state'],
            class_weight=self.config['class_weight'],
            n_jobs=-1
        )
        return model
    
    def create_risk_labels(self, features: pd.DataFrame) -> np.ndarray:
        """
        Create risk level labels from features
        
        Logic:
        - critical: days_of_supply < 3 OR stock_demand_ratio < 0.3
        - high: days_of_supply < 7 OR stock_demand_ratio < 0.5
        - medium: days_of_supply < 14 OR stock_demand_ratio < 0.8
        - low: otherwise
        """
        conditions = [
            (features['days_of_supply'] < 3) | (features['stock_demand_ratio'] < 0.3),
            (features['days_of_supply'] < 7) | (features['stock_demand_ratio'] < 0.5),
            (features['days_of_supply'] < 14) | (features['stock_demand_ratio'] < 0.8)
        ]
        
        labels = np.select(
            conditions,
            ['critical', 'high', 'medium'],
            default='low'
        )
        
        return labels
    
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the shortage detection model
        
        Args:
            X: Feature DataFrame
            y: Risk labels (if None, will be created from features)
            verbose: Verbosity level
        
        Returns:
            Training metrics dict
        """
        # Create labels if not provided
        if y is None:
            y = self.create_risk_labels(X)
        
        # Filter out non-numeric columns (like hospital_id, resource_type)
        # Only keep columns that can be converted to numeric
        numeric_columns = []
        for col in X.columns:
            if col in ['hospital_id', 'resource_type']:
                continue  # Skip metadata columns
            try:
                # Try to convert to numeric - if it fails, it's not numeric
                pd.to_numeric(X[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                # Skip non-numeric columns
                continue
        
        if not numeric_columns:
            raise ValueError("No numeric features found in input data. Please check your feature engineering.")
        
        # Select only numeric columns
        X_numeric = X[numeric_columns].copy()
        
        # Store feature names (only numeric ones)
        self.feature_names = numeric_columns
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Build and train model
        self.model = self.build_model()
        
        if verbose:
            print(f"\nTraining Shortage Detector")
            print(f"Features: {len(self.feature_names)}")
            print(f"Samples: {len(X)}")
            print(f"Class distribution:")
            unique, counts = np.unique(y, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
        
        self.model.fit(X_numeric, y_encoded)
        
        # Calculate training metrics
        y_pred = self.model.predict(X_numeric)
        accuracy = np.mean(y_pred == y_encoded)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics = {
            'accuracy': float(accuracy),
            'top_features': feature_importance.head(10).to_dict('records')
        }
        
        if verbose:
            print(f"\nTraining Accuracy: {accuracy:.4f}")
            print("\nTop 5 Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        self.save()
        
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        return_probabilities: bool = False
    ) -> np.ndarray:
        """
        Predict shortage risk levels
        
        Args:
            X: Feature DataFrame
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Risk level predictions or (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Ensure features match training (only numeric features)
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the features used during training
        X_numeric = X[self.feature_names].copy()
        
        # Ensure all columns are numeric
        for col in X_numeric.columns:
            if not pd.api.types.is_numeric_dtype(X_numeric[col]):
                try:
                    X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
                except (ValueError, TypeError):
                    raise ValueError(f"Column '{col}' cannot be converted to numeric. Please check your data.")
        
        # Predict
        y_pred_encoded = self.model.predict(X_numeric)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X_numeric)
            return y_pred, probabilities
        
        return y_pred
    
    def detect_shortages(
        self,
        current_inventory: pd.DataFrame,
        demand_predictions: pd.DataFrame,
        admissions_history: pd.DataFrame,
        hospital_info: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect shortages for all hospitals
        
        Returns:
            DataFrame with hospital_id, resource_type, risk_level, probability
        """
        # Engineer features
        features = engineer_shortage_features(
            current_inventory=current_inventory,
            demand_predictions=demand_predictions,
            admissions_history=admissions_history,
            hospital_info=hospital_info
        )
        
        # Keep only feature columns
        X = features[self.feature_names]
        
        # Predict
        risk_levels, probabilities = self.predict(X, return_probabilities=True)
        
        # Create result DataFrame
        results = pd.DataFrame({
            'hospital_id': features['hospital_id'],
            'resource_type': features['resource_type'],
            'risk_level': risk_levels,
            'confidence': probabilities.max(axis=1),
            'days_of_supply': features['days_of_supply'],
            'current_stock': features['stock_level'],
            'predicted_demand_7d': features['predicted_demand_7d']
        })
        
        # Add individual class probabilities
        for idx, risk_class in enumerate(self.label_encoder.classes_):
            results[f'prob_{risk_class}'] = probabilities[:, idx]
        
        return results
    
    def get_shortage_summary(
        self,
        detection_results: pd.DataFrame
    ) -> Dict:
        """
        Create summary of shortage detections
        
        Returns:
            Dict with summary statistics
        """
        summary = {
            'total_hospitals': len(detection_results),
            'by_risk_level': detection_results['risk_level'].value_counts().to_dict(),
            'critical_hospitals': detection_results[
                detection_results['risk_level'] == 'critical'
            ][['hospital_id', 'resource_type', 'days_of_supply']].to_dict('records'),
            'high_risk_hospitals': detection_results[
                detection_results['risk_level'] == 'high'
            ][['hospital_id', 'resource_type', 'days_of_supply']].to_dict('records'),
            'avg_days_of_supply_by_risk': detection_results.groupby('risk_level')['days_of_supply'].mean().to_dict()
        }
        
        return summary
    
    def save(self):
        """Save model, encoder, and feature names"""
        joblib.dump(self.model, str(self.model_path / 'model.pkl'))
        joblib.dump(self.label_encoder, str(self.model_path / 'label_encoder.pkl'))
        
        # Save feature names and config
        config = {
            'feature_names': self.feature_names,
            'risk_levels': self.config['risk_levels'],
            'n_estimators': self.config['n_estimators']
        }
        with open(self.model_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Shortage detector saved to {self.model_path}")
    
    def load(self):
        """Load trained model"""
        model_file = self.model_path / 'model.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found at {model_file}")
        
        self.model = joblib.load(str(model_file))
        self.label_encoder = joblib.load(str(self.model_path / 'label_encoder.pkl'))
        
        # Load config
        with open(self.model_path / 'config.json', 'r') as f:
            config = json.load(f)
        self.feature_names = config['feature_names']
        
        print(f"✓ Shortage detector loaded from {self.model_path}")
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model performance
        
        Returns:
            Dict with evaluation metrics
        """
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Predict
        y_pred = self.predict(X_test)
        
        # Encode for sklearn metrics
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        # Classification report
        report = classification_report(
            y_test_encoded,
            y_pred_encoded,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        metrics = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'overall_accuracy': float(report['accuracy']),
            'weighted_f1': float(report['weighted avg']['f1-score'])
        }
        
        return metrics