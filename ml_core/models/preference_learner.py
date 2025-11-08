"""
Preference Learning System - Hybrid ML + LLM
Learns user preferences from interactions and adapts recommendations

NOW WITH:
- Groq/Llama 3.3 70B for deep semantic analysis
- Qdrant vector store for similarity matching
- Hybrid scoring: 40% RF + 30% LLM + 30% Vector
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

# Handle both relative and absolute imports
try:
    from ..config import PREFERENCE_LEARNING_CONFIG, MODEL_PATHS
    from ..utils.feature_engineering import extract_recommendation_features
    from ..utils.groq_client import GroqPreferenceAnalyzer
    from ..utils.qdrant_client import InteractionVectorStore
except ImportError:
    from config import PREFERENCE_LEARNING_CONFIG, MODEL_PATHS
    from utils.feature_engineering import extract_recommendation_features
    try:
        from utils.groq_client import GroqPreferenceAnalyzer
    except ImportError:
        GroqPreferenceAnalyzer = None
    try:
        from utils.qdrant_client import InteractionVectorStore
    except ImportError:
        InteractionVectorStore = None


class PreferenceLearner:
    """Hybrid preference learning system with RF + LLM + Vector Store"""

    def __init__(self, use_llm=True, use_vector_store=True):
        """
        Initialize preference learner with optional LLM and vector store

        Args:
            use_llm: Enable Groq/Llama analysis (requires GROQ_API_KEY)
            use_vector_store: Enable Qdrant storage (requires Qdrant server)
        """
        self.config = PREFERENCE_LEARNING_CONFIG
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_path = MODEL_PATHS["preference_learner"]
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Default preference weights
        self.default_weights = self.config['feature_weights'].copy()

        # Initialize Groq/Llama analyzer
        self.use_llm = use_llm
        self.llm_analyzer = None
        if use_llm and GroqPreferenceAnalyzer is not None:
            try:
                self.llm_analyzer = GroqPreferenceAnalyzer()
            except Exception as e:
                print(f"⚠ Groq initialization failed: {e}")
                print("  Continuing without LLM analysis")
                self.use_llm = False
        else:
            self.use_llm = False
            if use_llm:
                print("⚠ Groq client not available. Install with: pip install groq")

        # Initialize Qdrant vector store
        self.use_vector_store = use_vector_store
        self.vector_store = None
        if use_vector_store and InteractionVectorStore is not None:
            try:
                self.vector_store = InteractionVectorStore()
            except Exception as e:
                print(f"⚠ Qdrant initialization failed: {e}")
                print("  Continuing without vector store")
                self.use_vector_store = False
        else:
            self.use_vector_store = False
            if use_vector_store:
                print("⚠ Qdrant client not available. Install with: pip install qdrant-client")
    
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
            # Safely get recommendations - handle both list and JSON string formats
            recs = interaction.get('recommendations', [])
            if isinstance(recs, str):
                import json
                try:
                    recs = json.loads(recs)
                except:
                    continue
            
            if not isinstance(recs, list) or not recs:
                continue
            
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
        
        # Calculate statistics (safely handle None values from database)
        response_times = []
        for i in interactions:
            rt = i.get('response_time_seconds')
            if rt is not None:
                response_times.append(rt)
        
        feedback_ratings = []
        for i in interactions:
            fr = i.get('feedback_rating')
            if fr is not None:
                feedback_ratings.append(fr)
        
        # Safely count recommendations
        total_recommendations = 0
        for i in interactions:
            recs = i.get('recommendations', [])
            if isinstance(recs, list):
                total_recommendations += len(recs)
        
        profile = {
            'user_id': user_id,
            'interaction_count': len(interactions),
            'preference_weights': preferences,
            'statistics': {
                'avg_response_time': np.mean(response_times) if response_times else 0,
                'avg_feedback_rating': np.mean(feedback_ratings) if feedback_ratings else 0,
                'total_recommendations_viewed': total_recommendations
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

    # =================================================================
    # HYBRID SCORING METHODS (Groq + Qdrant Integration)
    # =================================================================

    def update_from_interaction_enhanced(
        self,
        user_id: str,
        interaction: Dict,
        learning_rate: float = 0.1
    ):
        """
        Enhanced online learning with vector storage

        Combines:
        1. Random Forest update (existing)
        2. Vector storage in Qdrant

        Args:
            user_id: User identifier
            interaction: Interaction dict
            learning_rate: Learning rate (not used in RF, kept for compatibility)
        """
        # Store in Qdrant vector database
        if self.use_vector_store and self.vector_store:
            try:
                self.vector_store.store_interaction(user_id, interaction)
            except Exception as e:
                print(f"⚠ Vector storage failed: {e}")

        # Update Random Forest (existing logic)
        self.update_from_interaction(interaction, learning_rate)

    def score_recommendations_hybrid(
        self,
        user_id: str,
        recommendations: List[Dict],
        user_profile: Optional[Dict] = None
    ) -> List[Dict]:
        """
        HYBRID SCORING: Random Forest + LLM + Vector Similarity

        Scoring weights:
        - 40% Random Forest (fast pattern matching)
        - 30% LLM analysis (deep semantic understanding)
        - 30% Vector similarity (similar past decisions)

        Args:
            user_id: User identifier
            recommendations: List of recommendation dicts
            user_profile: Optional user profile (will use LLM insights)

        Returns:
            Recommendations ranked by hybrid score
        """
        if not recommendations:
            return []

        # Check if user has interaction history
        user_has_history = user_profile is not None and user_profile.get('interaction_count', 0) > 0

        # 1. Random Forest scoring (with cold start detection)
        rf_scores = self._score_with_rf(recommendations, user_has_history=user_has_history)

        # 2. LLM-based scoring (if enabled)
        llm_scores = self._score_with_llm(user_profile, recommendations) if self.use_llm else None

        # 3. Vector similarity scoring (if enabled)
        vector_scores = self._score_with_vectors(user_id, recommendations) if self.use_vector_store else None

        # 4. Combine scores with weights
        for idx, rec in enumerate(recommendations):
            scores = []
            weights = []

            # RF score (always available)
            scores.append(rf_scores[idx])
            weights.append(0.4)

            # LLM score
            if llm_scores:
                scores.append(llm_scores[idx])
                weights.append(0.3)

            # Vector score
            if vector_scores:
                scores.append(vector_scores[idx])
                weights.append(0.3)

            # Normalize weights to sum to 1.0
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # Weighted average
            final_score = sum(s * w for s, w in zip(scores, weights))

            rec['preference_score'] = float(final_score)
            rec['score_breakdown'] = {
                'rf_score': float(rf_scores[idx]),
                'llm_score': float(llm_scores[idx]) if llm_scores else None,
                'vector_score': float(vector_scores[idx]) if vector_scores else None,
                'weights': {
                    'rf': weights[0],
                    'llm': weights[1] if len(weights) > 1 else 0,
                    'vector': weights[2] if len(weights) > 2 else 0
                }
            }

        # Re-rank by preference score
        recommendations.sort(key=lambda x: x['preference_score'], reverse=True)

        return recommendations

    def _score_with_rf(self, recommendations: List[Dict], user_has_history: bool = True) -> List[float]:
        """Score recommendations using Random Forest"""
        # If user has no history, return neutral scores (true cold start)
        if not user_has_history:
            return [0.5] * len(recommendations)
        
        if self.model is None:
            # No model trained yet, return neutral scores
            return [0.5] * len(recommendations)

        try:
            # Extract features
            features_list = [extract_recommendation_features(rec) for rec in recommendations]
            X = pd.DataFrame(features_list)

            # Ensure feature order matches training
            X = X[self.feature_names]

            # Scale and predict
            X_scaled = self.scaler.transform(X)
            scores = self.model.predict(X_scaled)

            # Clip to 0-1 range
            scores = np.clip(scores, 0, 1)

            return scores.tolist()

        except Exception as e:
            print(f"⚠ RF scoring failed: {e}")
            return [0.5] * len(recommendations)

    def _score_with_llm(
        self,
        user_profile: Optional[Dict],
        recommendations: List[Dict]
    ) -> Optional[List[float]]:
        """Score recommendations using Groq/Llama LLM"""
        if not self.llm_analyzer or not user_profile:
            return None

        scores = []

        for rec in recommendations:
            try:
                # Get LLM explanation
                explanation = self.llm_analyzer.explain_recommendation_ranking(
                    user_profile, rec
                )

                # Simple heuristic: longer explanations = better fit
                # In production, could ask LLM to return score directly
                score = min(len(explanation) / 200, 1.0)
                scores.append(score)

                # Store explanation in recommendation
                rec['llm_explanation'] = explanation

            except Exception as e:
                print(f"⚠ LLM scoring error: {e}")
                scores.append(0.5)

        return scores

    def _score_with_vectors(
        self,
        user_id: str,
        recommendations: List[Dict]
    ) -> Optional[List[float]]:
        """Score based on similarity to past successful choices"""
        if not self.vector_store:
            return None

        scores = []

        for rec in recommendations:
            try:
                # Create mock interaction for this recommendation
                mock_interaction = {
                    'selected_recommendation_index': 0,
                    'recommendations': [rec]
                }

                # Find similar past interactions
                similar = self.vector_store.find_similar_interactions(
                    interaction=mock_interaction,
                    user_id=user_id,
                    limit=5
                )

                # Average similarity score
                if similar:
                    avg_similarity = float(np.mean([s['score'] for s in similar]))
                    scores.append(avg_similarity)
                else:
                    # No history, neutral score
                    scores.append(0.5)

            except Exception as e:
                print(f"⚠ Vector scoring error: {e}")
                scores.append(0.5)

        return scores

    def get_user_profile_enhanced(
        self,
        user_id: str,
        interactions: List[Dict]
    ) -> Dict:
        """
        Enhanced user profile with LLM insights and vector stats

        Returns:
            Dict with:
            - Basic profile (from base class)
            - LLM-analyzed preference patterns
            - Vector store statistics
        """
        # Base profile
        profile = self.get_user_profile(user_id, interactions)

        # Add LLM analysis (if enough interactions)
        if self.use_llm and self.llm_analyzer and len(interactions) >= 3:
            try:
                llm_insights = self.llm_analyzer.analyze_interaction_pattern(interactions)
                profile['llm_insights'] = llm_insights
                profile['preference_type'] = llm_insights.get('preference_type', 'balanced')
                profile['key_patterns'] = llm_insights.get('key_patterns', [])

                # Use LLM-derived objective weights if available
                if 'objective_weights' in llm_insights:
                    profile['objective_weights'] = llm_insights['objective_weights']

            except Exception as e:
                print(f"⚠ LLM profile analysis failed: {e}")

        # Add vector store statistics
        if self.use_vector_store and self.vector_store:
            try:
                count = self.vector_store.get_user_interaction_count(user_id)
                stats = self.vector_store.get_collection_stats()
                profile['vector_store'] = {
                    'user_interaction_count': count,
                    'total_interactions': stats.get('total_points', 0)
                }
            except Exception as e:
                print(f"⚠ Vector store query failed: {e}")

        return profile