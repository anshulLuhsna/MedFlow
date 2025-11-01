"""
Unified ML Core Interface
Single entry point for all ML functionality used by agent frameworks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .models.demand_forecaster import DemandForecaster
from .models.shortage_detector import ShortageDetector
from .models.optimizer import ResourceOptimizer
from .models.preference_learner import PreferenceLearner
from .utils.data_loader import DataLoader
from .config import RESOURCE_TYPES


class MLCore:
    """
    Unified ML Core - Single interface for all ML operations
    
    This class is used by both LangGraph and CrewAI agent frameworks
    """
    
    def __init__(self, supabase_client=None):
        """
        Initialize ML Core with all models
        
        Args:
            supabase_client: Optional Supabase client (will create if not provided)
        """
        self.data_loader = DataLoader(supabase_client)
        
        # Initialize models for each resource type
        self.demand_forecasters = {
            resource_type: DemandForecaster(resource_type)
            for resource_type in RESOURCE_TYPES
        }
        
        self.shortage_detector = ShortageDetector()
        self.optimizer = ResourceOptimizer()
        self.preference_learner = PreferenceLearner()
        
        # Load trained models
        self.load_models()
    
    def load_models(self, verbose: bool = False):
        """Load all trained models"""
        try:
            # Load demand forecasters
            for resource_type, forecaster in self.demand_forecasters.items():
                try:
                    forecaster.load()
                except FileNotFoundError:
                    if verbose:
                        print(f"⚠ Demand forecaster for {resource_type} not found")
            
            # Load shortage detector
            try:
                self.shortage_detector.load()
            except FileNotFoundError:
                if verbose:
                    print("⚠ Shortage detector not found")
            
            # Load preference learner
            self.preference_learner.load()
            
            if verbose:
                print("✓ ML Core models loaded")
        
        except Exception as e:
            if verbose:
                print(f"⚠ Error loading models: {e}")
    
    # ============================================
    # DEMAND FORECASTING
    # ============================================
    
    def predict_demand(
        self,
        hospital_id: str,
        resource_type: str,
        days_ahead: int = 14
    ) -> Dict:
        """
        Predict future resource demand for a specific hospital
        
        Args:
            hospital_id: Hospital UUID
            resource_type: Type of resource
            days_ahead: Forecast horizon (max 14 days)
        
        Returns:
            Dict with predictions and confidence intervals
        """
        if resource_type not in self.demand_forecasters:
            raise ValueError(f"Invalid resource type: {resource_type}")
        
        forecaster = self.demand_forecasters[resource_type]
        
        # Get historical data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=60)
        
        inventory_history = self.data_loader.get_inventory_history(
            start_date=str(start_date),
            end_date=str(end_date),
            hospital_ids=[hospital_id]
        )
        
        admissions_history = self.data_loader.get_admissions_history(
            start_date=str(start_date),
            end_date=str(end_date),
            hospital_ids=[hospital_id]
        )
        
        # Merge data
        data = inventory_history.merge(
            admissions_history,
            left_on=['hospital_id', 'date'],
            right_on=['hospital_id', 'admission_date'],
            how='left'
        )
        
        # Predict
        prediction = forecaster.predict_for_hospital(
            hospital_id=hospital_id,
            historical_data=data,
            days_ahead=days_ahead
        )
        
        return prediction
    
    def predict_demand_all_hospitals(
        self,
        resource_type: str,
        days_ahead: int = 14
    ) -> pd.DataFrame:
        """
        Predict demand for all hospitals
        
        Returns:
            DataFrame with predictions for all hospitals
        """
        hospitals = self.data_loader.get_hospitals()
        
        predictions = []
        for _, hospital in hospitals.iterrows():
            try:
                pred = self.predict_demand(
                    hospital_id=hospital['id'],
                    resource_type=resource_type,
                    days_ahead=days_ahead
                )
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting for hospital {hospital['id']}: {e}")
        
        return pd.DataFrame(predictions)
    
    # ============================================
    # SHORTAGE DETECTION
    # ============================================
    
    def detect_shortages(
        self,
        resource_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect shortage risks across all hospitals
        
        Args:
            resource_type: Optional filter for specific resource type
        
        Returns:
            DataFrame with shortage predictions
        """
        # Get current inventory
        current_inventory = self.data_loader.get_current_inventory()
        
        # Get demand predictions
        if resource_type:
            demand_preds = self.predict_demand_all_hospitals(resource_type)
        else:
            # Get for all resource types
            all_preds = []
            for rt in RESOURCE_TYPES:
                preds = self.predict_demand_all_hospitals(rt)
                all_preds.append(preds)
            demand_preds = pd.concat(all_preds, ignore_index=True)
        
        # Get recent admissions
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        admissions = self.data_loader.get_admissions_history(
            start_date=str(start_date),
            end_date=str(end_date)
        )
        
        # Get hospital info
        hospitals = self.data_loader.get_hospitals()
        
        # Detect shortages
        results = self.shortage_detector.detect_shortages(
            current_inventory=current_inventory,
            demand_predictions=demand_preds,
            admissions_history=admissions,
            hospital_info=hospitals
        )
        
        return results
    
    def get_shortage_summary(self) -> Dict:
        """
        Get summary of current shortage situation
        
        Returns:
            Dict with aggregated shortage statistics
        """
        shortages = self.detect_shortages()
        summary = self.shortage_detector.get_shortage_summary(shortages)
        return summary
    
    # ============================================
    # OPTIMIZATION
    # ============================================
    
    def optimize_allocation(
        self,
        resource_type: str,
        shortage_hospital_ids: List[str] = None,
        objective_weights: Dict[str, float] = None
    ) -> Dict:
        """
        Generate optimal resource allocation strategy
        
        Args:
            resource_type: Type of resource to allocate
            shortage_hospital_ids: Optional list of hospitals with shortages
            objective_weights: Custom optimization objectives
        
        Returns:
            Dict with optimal allocation strategy
        """
        # Detect shortages if not provided
        if shortage_hospital_ids is None:
            all_shortages = self.detect_shortages(resource_type)
            shortage_hospitals = all_shortages[
                all_shortages['risk_level'].isin(['high', 'critical'])
            ]
        else:
            shortage_hospitals = self.detect_shortages(resource_type)
            shortage_hospitals = shortage_hospitals[
                shortage_hospitals['hospital_id'].isin(shortage_hospital_ids)
            ]
        
        # Find hospitals with surplus
        current_inventory = self.data_loader.get_current_inventory()
        resource_inventory = current_inventory[
            current_inventory['resource_type'] == resource_type
        ]
        
        # Surplus = available quantity > 2x critical threshold
        resource_types = self.data_loader.get_resource_types()
        threshold = resource_types[
            resource_types['name'] == resource_type
        ]['critical_threshold'].values[0]
        
        surplus_hospitals = resource_inventory[
            resource_inventory['available_quantity'] > threshold * 2
        ]
        
        # Get hospital info
        hospitals = self.data_loader.get_hospitals()
        
        # Optimize
        result = self.optimizer.optimize_allocation(
            shortage_hospitals=shortage_hospitals,
            surplus_hospitals=surplus_hospitals,
            hospital_info=hospitals,
            resource_type=resource_type,
            objective_weights=objective_weights
        )
        
        return result
    
    def generate_allocation_strategies(
        self,
        resource_type: str,
        n_strategies: int = 3
    ) -> List[Dict]:
        """
        Generate multiple allocation strategies
        
        Returns:
            List of ranked strategies
        """
        # Detect shortages
        all_shortages = self.detect_shortages(resource_type)
        shortage_hospitals = all_shortages[
            all_shortages['risk_level'].isin(['high', 'critical'])
        ]
        
        # Find surplus
        current_inventory = self.data_loader.get_current_inventory()
        resource_inventory = current_inventory[
            current_inventory['resource_type'] == resource_type
        ]
        
        resource_types = self.data_loader.get_resource_types()
        threshold = resource_types[
            resource_types['name'] == resource_type
        ]['critical_threshold'].values[0]
        
        surplus_hospitals = resource_inventory[
            resource_inventory['available_quantity'] > threshold * 2
        ]
        
        # Get hospital info
        hospitals = self.data_loader.get_hospitals()
        
        # Generate strategies
        strategies = self.optimizer.generate_multiple_strategies(
            shortage_hospitals=shortage_hospitals,
            surplus_hospitals=surplus_hospitals,
            hospital_info=hospitals,
            resource_type=resource_type,
            n_strategies=n_strategies
        )
        
        return strategies
    
    # ============================================
    # PREFERENCE LEARNING
    # ============================================
    
    def update_preferences(
        self,
        user_id: str,
        interaction: Dict
    ):
        """
        Update user preferences from interaction
        
        Args:
            user_id: User identifier
            interaction: Dict with interaction details
        """
        self.preference_learner.update_from_interaction(interaction)
        
        # Store in database (you could also store in Qdrant here)
        # For now, just update the model
    
    def score_recommendations(
        self,
        recommendations: List[Dict],
        user_id: str = None
    ) -> List[Dict]:
        """
        Score and re-rank recommendations based on user preferences
        
        Args:
            recommendations: List of recommendation dicts
            user_id: Optional user ID for personalization
        
        Returns:
            Recommendations ranked by preference
        """
        # Get user profile if available
        user_profile = None
        if user_id:
            # Load from database
            pass
        
        return self.preference_learner.score_recommendations(
            recommendations=recommendations,
            user_profile=user_profile
        )
    
    def get_user_preferences(
        self,
        user_id: str,
        interactions: List[Dict]
    ) -> Dict:
        """
        Get comprehensive user preference profile
        
        Returns:
            Dict with preference weights and insights
        """
        return self.preference_learner.get_user_profile(user_id, interactions)
    
    # ============================================
    # UTILITY FUNCTIONS
    # ============================================
    
    def get_hospital_status(
        self,
        hospital_id: str
    ) -> Dict:
        """
        Get complete status for a hospital including predictions
        
        Returns:
            Dict with current state, predictions, and risk assessment
        """
        # Current inventory
        inventory = self.data_loader.get_current_inventory()
        hospital_inventory = inventory[inventory['hospital_id'] == hospital_id]
        
        # Predictions for each resource
        predictions = {}
        for resource_type in RESOURCE_TYPES:
            try:
                pred = self.predict_demand(hospital_id, resource_type)
                predictions[resource_type] = pred
            except:
                predictions[resource_type] = None
        
        # Shortage risks
        shortages = self.detect_shortages()
        hospital_shortages = shortages[shortages['hospital_id'] == hospital_id]
        
        status = {
            'hospital_id': hospital_id,
            'current_inventory': hospital_inventory.to_dict('records'),
            'demand_predictions': predictions,
            'shortage_risks': hospital_shortages.to_dict('records'),
            'timestamp': datetime.now().isoformat()
        }
        
        return status