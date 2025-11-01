"""
Preference Learning System - Hybrid ML + LLM
Learns user preferences from interactions and adapts recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

# Handle both relative and absolute imports
try:
from ..config import PREFERENCE_LEARNING_CONFIG, MODEL_PATHS
from ..utils.feature_engineering import extract_recommendation_features
except ImportError:
    from config import PREFERENCE_LEARNING_CONFIG, MODEL_PATHS
    from utils.feature_engineering import extract_recommendation_features


class PreferenceLearner:
    """Hybrid preference learning system"""
    
    def __init__(self):
        self.config = PREFERENCE_LEARNING_CONFIG
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_path = MODEL_PATHS["preference_learner"]
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Default preference weights
        self.default_weights = self.config['feature_weights'].copy()
    
    def build_model(self) -> RandomForestRegressor:
        """Build preference prediction model"""
        model = RandomForestRegressor(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        return model
    
    def extract_features_from_interaction(
        self,
        interaction: Dict
    ) -> pd.DataFrame:
        """
        Extract features from user interaction
        
        Args:
            interaction: Dict containing:
                - recommendations: List of recommendation dicts
                - selected_index: Which recommendation was chosen
        
        Returns:
            DataFrame with features and labels
        """
        recommendations = interaction['recommendations']
        selected_index = interaction.get('selected_recommendation_index', 0)
        
        features_list = []
        labels = []
        
        for idx, rec in enumerate(recommendations):
            # Extract features
            features = extract_recommendation_features(rec)
            features_list.append(features)
            
            # Label: 1 if selected, 0 if not
            labels.append(1.0 if idx == selected_index else 0.0)
        
        df = pd.DataFrame(features_list)
        df['label'] = labels
        
        return df
    
    def train(
        self,
        interactions: List[Dict],
        verbose: int = 1
    ) -> Dict:
        """
        Train preference model from user interactions
        
        Args:
            interactions: List of interaction dicts
            verbose: Verbosity level
        
        Returns:
            Training metrics
        """
        # Extract features from all interactions
        all_features = []
        all_labels = []
        
        for interaction in interactions:
            df = self.extract_features_from_interaction(interaction)
            all_features.append(df.drop('label', axis=1))
            all_labels.extend(df['label'].tolist())
        
        if not all_features:
            raise ValueError("No training data available")
        
        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_labels)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build and train model
        self.model = self.build_model()
        
        if verbose:
            print(f"\nTraining Preference Learner")
            print(f"Interactions: {len(interactions)}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Samples: {len(X)}")
            print(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        
        self.model.fit(X_scaled, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate training score
        y_pred = self.model.predict(X_scaled)
        mse = np.mean((y_pred - y) ** 2)
        
        metrics = {
            'mse': float(mse),
            'top_features': feature_importance.head(10).to_dict('records')
        }
        
        if verbose:
            print(f"\nTraining MSE: {mse:.4f}")
            print("\nTop 5 Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        self.save()
        
        return metrics
    
    def update_from_interaction(
        self,
        interaction: Dict,
        learning_rate: float = 0.1
    ):
        """
        Online learning: Update model from single interaction
        
        Args:
            interaction: Single interaction dict
            learning_rate: How much to weight this new interaction
        """
        if self.model is None:
            # Initialize with this interaction
            self.train([interaction], verbose=0)
            return
        
        # Extract features
        df = self.extract_features_from_interaction(interaction)
        X = df.drop('label', axis=1)
        y = df['label'].values
        
        # Ensure feature order matches
        X = X[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Get current predictions
        current_pred = self.model.predict(X_scaled)
        
        # Partial update using warm_start (simplified online learning)
        # In production, you might use more sophisticated online learning
        self.model.fit(X_scaled, y)
    
    def score_recommendations(
        self,
        recommendations: List[Dict],
        user_profile: Dict = None
    ) -> List[Dict]:
        """
        Score recommendations based on learned preferences
        
        Args:
            recommendations: List of recommendation dicts
            user_profile: Optional user preference profile
        
        Returns:
            Recommendations with preference scores added
        """
        if self.model is None or len(recommendations) == 0:
            # No model trained, return original scores
            for rec in recommendations:
                rec['preference_score'] = 0.5
                rec['adjusted_score'] = rec.get('overall_score', 0.5)
            return recommendations
        
        # Extract features
        features_list = [extract_recommendation_features(rec) for rec in recommendations]
        X = pd.DataFrame(features_list)
        
        # Ensure feature order
        X = X[self.feature_names]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        preference_scores = self.model.predict(X_scaled)
        
        # Normalize scores to 0-1
        if len(preference_scores) > 1:
            min_score = preference_scores.min()
            max_score = preference_scores.max()
            if max_score > min_score:
                preference_scores = (preference_scores - min_score) / (max_score - min_score)
        
        # Add scores to recommendations
        for rec, pref_score in zip(recommendations, preference_scores):
            rec['preference_score'] = float(pref_score)
            
            # Adjust overall score
            original_score = rec.get('overall_score', 0.5)
            rec['adjusted_score'] = 0.6 * original_score + 0.4 * pref_score
        
        # Re-rank by adjusted score
        recommendations.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        return recommendations
    
    def extract_implicit_preferences(
        self,
        interactions: List[Dict],
        use_llm: bool = True
    ) -> Dict:
        """
        Analyze interactions to extract implicit preference patterns
        
        Args:
            interactions: List of user interactions
            use_llm: Whether to use LLM for deeper analysis
        
        Returns:
            Dict with preference insights
        """
        if not interactions:
            return self.default_weights.copy()
        
        # Statistical analysis
        selected_features = []
        
        for interaction in interactions:
            recs = interaction['recommendations']
            selected_idx = interaction.get('selected_recommendation_index', 0)
            
            if selected_idx < len(recs):
                selected = recs[selected_idx]
                features = extract_recommendation_features(selected)
                selected_features.append(features)
        
        if not selected_features:
            return self.default_weights.copy()
        
        df = pd.DataFrame(selected_features)
        
        # Calculate averages of what user tends to select
        preferences = {
            'cost_preference': 1 - df['cost_score'].mean(),  # Lower cost score = prefers low cost
            'speed_preference': df['speed_score'].mean(),
            'coverage_preference': df['coverage_score'].mean(),
            'fairness_preference': df['fairness_score'].mean()
        }
        
        # Normalize to sum to 1
        total = sum(preferences.values())
        if total > 0:
            preferences = {k: v/total for k, v in preferences.items()}
        
        # LLM-based analysis (optional, more sophisticated)
        if use_llm:
            llm_insights = self._llm_analyze_preferences(interactions)
            preferences['llm_insights'] = llm_insights
        
        return preferences
    
    def _llm_analyze_preferences(self, interactions: List[Dict]) -> Dict:
        """
        Use LLM to analyze preference patterns (placeholder for actual implementation)
        
        In production, this would call Claude API with interaction history
        """
        # Placeholder - implement actual LLM call when integrating with agents
        return {
            'pattern_detected': 'cost_conscious',
            'confidence': 0.75,
            'reasoning': 'User consistently chooses lower-cost options'
        }
    
    def get_user_profile(self, user_id: str, interactions: List[Dict]) -> Dict:
        """
        Create comprehensive user preference profile
        
        Returns:
            Dict with preference weights, statistics, and insights
        """
        if not interactions:
            return {
                'user_id': user_id,
                'interaction_count': 0,
                'preference_weights': self.default_weights.copy(),
                'initialized': False
            }
        
        # Extract preferences
        preferences = self.extract_implicit_preferences(interactions)
        
        # Calculate statistics
        response_times = [
            i.get('response_time_seconds', 0)
            for i in interactions
        ]
        
        feedback_ratings = [
            i.get('feedback_rating', 0)
            for i in interactions
            if i.get('feedback_rating') is not None
        ]
        
        profile = {
            'user_id': user_id,
            'interaction_count': len(interactions),
            'preference_weights': preferences,
            'statistics': {
                'avg_response_time': np.mean(response_times) if response_times else 0,
                'avg_feedback_rating': np.mean(feedback_ratings) if feedback_ratings else 0,
                'total_recommendations_viewed': sum([
                    len(i['recommendations']) for i in interactions
                ])
            },
            'initialized': True,
            'last_updated': datetime.now().isoformat()
        }
        
        return profile
    
    def save(self):
        """Save model and scaler"""
        if self.model is not None:
            joblib.dump(self.model, str(self.model_path / 'model.pkl'))
            joblib.dump(self.scaler, str(self.model_path / 'scaler.pkl'))
            
            config = {
                'feature_names': self.feature_names,
                'default_weights': self.default_weights
            }
            with open(self.model_path / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✓ Preference learner saved to {self.model_path}")
    
    def load(self):
        """Load trained model"""
        model_file = self.model_path / 'model.pkl'
        
        if not model_file.exists():
            print(f"⚠ No trained preference model found at {model_file}")
            print("  Will initialize on first interaction")
            return
        
        self.model = joblib.load(str(model_file))
        self.scaler = joblib.load(str(self.model_path / 'scaler.pkl'))
        
        with open(self.model_path / 'config.json', 'r') as f:
            config = json.load(f)
        self.feature_names = config['feature_names']
        
        print(f"✓ Preference learner loaded from {self.model_path}")