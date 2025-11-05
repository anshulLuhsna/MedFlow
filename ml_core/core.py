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

        # Initialize with hybrid features (Groq LLM + Qdrant Vector Store)
        self.preference_learner = PreferenceLearner(
            use_llm=True,
            use_vector_store=True
        )

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
        # Fetch data from 2023 onwards (for historical datasets)
        # This ensures we get all available historical data
        start_date = datetime(2023, 1, 1).date()
        
        inventory_history = self.data_loader.get_inventory_history(
            start_date=str(start_date),
            end_date=None,  # Don't limit end_date, get all available data up to today
            hospital_ids=[hospital_id]
        )
        
        admissions_history = self.data_loader.get_admissions_history(
            start_date=str(start_date),
            end_date=None,  # Don't limit end_date, get all available data up to today
            hospital_ids=[hospital_id]
        )
        
        # Check if inventory_history is empty or missing date column
        if inventory_history.empty:
            raise ValueError(
                f"No inventory history found for hospital {hospital_id} "
                f"from {start_date} onwards"
            )
        
        if 'date' not in inventory_history.columns:
            # Check for alternative date column names
            date_cols = [col for col in inventory_history.columns if 'date' in col.lower()]
            if date_cols:
                raise ValueError(
                    f"inventory_history missing 'date' column. Found columns: {inventory_history.columns.tolist()}. "
                    f"Date-like columns: {date_cols}"
                )
            else:
                raise ValueError(
                    f"inventory_history must contain 'date' column. "
                    f"Available columns: {inventory_history.columns.tolist()}"
                )
        
        # Filter for the specific resource type
        resource_types = self.data_loader.get_resource_types()
        if resource_types.empty:
            raise ValueError(f"Resource types not found in database")
        
        resource_type_match = resource_types[resource_types['name'] == resource_type]
        if resource_type_match.empty:
            raise ValueError(
                f"Resource type '{resource_type}' not found. "
                f"Available types: {resource_types['name'].tolist()}"
            )
        
        resource_id = resource_type_match['id'].values[0]
        inventory_history = inventory_history[inventory_history['resource_type_id'] == resource_id]
        
        # Check if we have data after filtering
        if inventory_history.empty:
            raise ValueError(
                f"No inventory history found for hospital {hospital_id} "
                f"and resource type '{resource_type}' (ID: {resource_id}) "
                f"from {start_date} onwards"
            )
        
        # Ensure resupply column exists (calculate from quantity changes if missing)
        if 'resupply' not in inventory_history.columns:
            inventory_history['resupply'] = 0
            # Calculate resupply: quantity[t] = quantity[t-1] - consumption[t-1] + resupply[t]
            # So resupply[t] = quantity[t] - quantity[t-1] + consumption[t-1]
            if len(inventory_history) > 1:
                inventory_history = inventory_history.sort_values('date')
                quantity_change = inventory_history['quantity'].diff()
                consumption_prev = inventory_history['consumption'].shift(1).fillna(0)
                # Resupply = quantity_change + previous_consumption (must be non-negative)
                inventory_history['resupply'] = (quantity_change + consumption_prev).clip(lower=0)
        
        # Merge data
        data = inventory_history.merge(
            admissions_history,
            left_on=['hospital_id', 'date'],
            right_on=['hospital_id', 'admission_date'],
            how='left'
        )
        
        # Ensure total_admissions column exists
        if 'total_admissions' not in data.columns:
            data['total_admissions'] = 0
        
        # Use 'date' column consistently (remove admission_date if it exists separately)
        if 'admission_date' in data.columns and 'date' in data.columns:
            # Keep 'date' from inventory_history as primary
            data = data.drop(columns=['admission_date'], errors='ignore')
        
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
        days_ahead: int = 14,
        hospital_limit: int = None
    ) -> pd.DataFrame:
        """
        Predict demand for all hospitals
        
        Args:
            resource_type: Type of resource to predict
            days_ahead: Number of days to forecast
            hospital_limit: Optional limit on number of hospitals to process (max 100)
        
        Returns:
            DataFrame with predictions for all hospitals
        """
        hospitals = self.data_loader.get_hospitals(limit=hospital_limit)
        
        # Check if hospitals DataFrame is empty
        if hospitals.empty:
            print(f"[Predict Demand] No hospitals found for resource type {resource_type}")
            return pd.DataFrame()
        
        print(f"[Predict Demand] Predicting demand for {len(hospitals)} hospitals (resource: {resource_type})")
        
        predictions = []
        successful = 0
        failed = 0
        
        for idx, (_, hospital) in enumerate(hospitals.iterrows()):
            try:
                # Use hospital_id (get_hospitals() renames 'id' to 'hospital_id')
                hospital_id = hospital.get('hospital_id') or hospital.get('id')
                if hospital_id is None:
                    failed += 1
                    continue
                
                # Progress logging for large datasets
                if (idx + 1) % 10 == 0:
                    print(f"[Predict Demand] Progress: {idx+1}/{len(hospitals)} hospitals processed")
                
                pred = self.predict_demand(
                    hospital_id=hospital_id,
                    resource_type=resource_type,
                    days_ahead=days_ahead
                )
                
                # Validate prediction structure
                if not isinstance(pred, dict):
                    print(f"[Predict Demand] Warning: Prediction for hospital {hospital_id} is not a dict: {type(pred)}")
                    failed += 1
                    continue
                
                if 'predicted_demand' not in pred:
                    print(f"[Predict Demand] Warning: Prediction for hospital {hospital_id} missing 'predicted_demand' key")
                    failed += 1
                    continue
                
                predictions.append(pred)
                successful += 1
            except Exception as e:
                failed += 1
                hospital_id = hospital.get('hospital_id') or hospital.get('id', 'unknown')
                # Only log first few errors to avoid spam
                if failed <= 3:
                    print(f"[Predict Demand] Error for hospital {hospital_id}: {e}")
        
        print(f"[Predict Demand] Completed: {successful} successful, {failed} failed")
        
        if not predictions:
            print(f"[Predict Demand] Warning: No predictions generated for resource type {resource_type}")
            return pd.DataFrame()
        
        # Convert predictions to DataFrame format expected by shortage_detector
        # Each prediction dict has: hospital_id, resource_type, forecast_dates, predicted_demand
        # Need to expand to: hospital_id, resource_type, day, predicted_consumption
        expanded_predictions = []
        for pred in predictions:
            hospital_id = pred.get('hospital_id')
            resource_type = pred.get('resource_type')
            predicted_demand = pred.get('predicted_demand', [])
            
            # Skip if no predictions
            if not predicted_demand or not isinstance(predicted_demand, list):
                continue
            
            # Expand each day's prediction into a separate row
            for day, consumption in enumerate(predicted_demand):
                expanded_predictions.append({
                    'hospital_id': str(hospital_id),  # Ensure string type
                    'resource_type': str(resource_type),  # Ensure string type
                    'day': int(day),  # Ensure integer type
                    'predicted_consumption': float(consumption) if consumption is not None else 0.0
                })
        
        if not expanded_predictions:
            print(f"[Predict Demand] Warning: No expanded predictions generated")
            return pd.DataFrame(columns=['hospital_id', 'resource_type', 'day', 'predicted_consumption'])
        
        df = pd.DataFrame(expanded_predictions)
        
        # Ensure required columns exist
        required_cols = ['hospital_id', 'resource_type', 'day', 'predicted_consumption']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[Predict Demand] Error: Missing columns in DataFrame: {missing_cols}")
            print(f"[Predict Demand] Available columns: {df.columns.tolist()}")
            return pd.DataFrame(columns=required_cols)
        
        print(f"[Predict Demand] Expanded to {len(df)} rows with columns: {df.columns.tolist()}")
        return df
    
    # ============================================
    # SHORTAGE DETECTION
    # ============================================
    
    def detect_shortages(
        self,
        resource_type: Optional[str] = None,
        hospital_limit: int = None
    ) -> pd.DataFrame:
        """
        Detect shortage risks across all hospitals
        
        Args:
            resource_type: Optional filter for specific resource type
        
        Returns:
            DataFrame with shortage predictions
        """
        try:
            print(f"[Shortage Detection] Starting shortage detection (resource_type={resource_type})")
            
            # Get current inventory
            print("[Shortage Detection] Fetching current inventory...")
            current_inventory = self.data_loader.get_current_inventory()
            if current_inventory.empty:
                print("[Shortage Detection] Warning: No current inventory data found")
                return pd.DataFrame()
            print(f"[Shortage Detection] Found {len(current_inventory)} inventory records")
            
            # Get demand predictions
            print("[Shortage Detection] Generating demand predictions...")
            if resource_type:
                print(f"[Shortage Detection] Predicting demand for resource type: {resource_type}")
                demand_preds = self.predict_demand_all_hospitals(resource_type, hospital_limit=hospital_limit)
                print(f"[Shortage Detection] Generated {len(demand_preds)} demand prediction rows")
                if not demand_preds.empty:
                    print(f"[Shortage Detection] Demand prediction columns: {demand_preds.columns.tolist()}")
                    print(f"[Shortage Detection] Sample demand prediction:\n{demand_preds.head()}")
            else:
                # Get for all resource types (but limit to avoid hanging)
                print(f"[Shortage Detection] Predicting demand for all resource types: {RESOURCE_TYPES}")
                all_preds = []
                for idx, rt in enumerate(RESOURCE_TYPES):
                    print(f"[Shortage Detection] Processing resource type {idx+1}/{len(RESOURCE_TYPES)}: {rt}")
                    try:
                        preds = self.predict_demand_all_hospitals(rt, hospital_limit=hospital_limit)
                        if not preds.empty:
                            all_preds.append(preds)
                            print(f"[Shortage Detection] Generated {len(preds)} predictions for {rt}")
                    except Exception as e:
                        print(f"[Shortage Detection] Error predicting demand for {rt}: {e}")
                        continue
                
                if not all_preds:
                    print("[Shortage Detection] Warning: No demand predictions generated")
                    return pd.DataFrame()
                
                demand_preds = pd.concat(all_preds, ignore_index=True)
                print(f"[Shortage Detection] Total demand predictions: {len(demand_preds)} rows")
                if not demand_preds.empty:
                    print(f"[Shortage Detection] Demand prediction columns: {demand_preds.columns.tolist()}")
                    print(f"[Shortage Detection] Sample demand prediction:\n{demand_preds.head()}")
                else:
                    print("[Shortage Detection] Warning: Empty demand predictions DataFrame")
                    return pd.DataFrame()
            
            # Get recent admissions (from 2023 onwards for historical data)
            print("[Shortage Detection] Fetching admissions history...")
            start_date = datetime(2023, 1, 1).date()
            admissions = self.data_loader.get_admissions_history(
                start_date=str(start_date),
                end_date=None  # Get all available data
            )
            print(f"[Shortage Detection] Found {len(admissions)} admissions records")
            
            # Get hospital info
            print("[Shortage Detection] Fetching hospital info...")
            hospitals = self.data_loader.get_hospitals(limit=hospital_limit)
            if hospitals.empty:
                print("[Shortage Detection] Warning: No hospital data found")
                return pd.DataFrame()
            print(f"[Shortage Detection] Found {len(hospitals)} hospitals")
            
            # Validate demand_predictions DataFrame before passing to shortage_detector
            if demand_preds.empty:
                print("[Shortage Detection] Error: Empty demand predictions DataFrame")
                return pd.DataFrame()
            
            required_cols = ['hospital_id', 'resource_type', 'day', 'predicted_consumption']
            missing_cols = [col for col in required_cols if col not in demand_preds.columns]
            if missing_cols:
                print(f"[Shortage Detection] Error: Missing required columns in demand_predictions: {missing_cols}")
                print(f"[Shortage Detection] Available columns: {demand_preds.columns.tolist()}")
                return pd.DataFrame()
            
            print(f"[Shortage Detection] Validated demand_predictions: {len(demand_preds)} rows, columns: {demand_preds.columns.tolist()}")
            
            # Detect shortages
            print("[Shortage Detection] Running shortage detection model...")
            results = self.shortage_detector.detect_shortages(
                current_inventory=current_inventory,
                demand_predictions=demand_preds,
                admissions_history=admissions,
                hospital_info=hospitals
            )
            
            print(f"[Shortage Detection] Completed. Found {len(results)} shortage predictions")
            return results
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[Shortage Detection] Error: {e}")
            print(f"[Shortage Detection] Traceback:\n{error_trace}")
            raise
    
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
        objective_weights: Dict[str, float] = None,
        hospital_limit: int = None
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
        print(f"[Optimize Allocation] Starting optimization for resource: {resource_type}")
        
        # Detect shortages if not provided
        if shortage_hospital_ids is None:
            all_shortages = self.detect_shortages(resource_type, hospital_limit=hospital_limit)
            # Include medium, high, and critical risk levels (not just high/critical)
            # This allows optimization to help hospitals before they reach critical status
            shortage_hospitals = all_shortages[
                all_shortages['risk_level'].isin(['medium', 'high', 'critical'])
            ].copy()
            print(f"[Optimize Allocation] Found {len(all_shortages)} total shortage predictions")
            print(f"[Optimize Allocation] Risk level breakdown: {all_shortages['risk_level'].value_counts().to_dict()}")
        else:
            shortage_hospitals = self.detect_shortages(resource_type, hospital_limit=hospital_limit)
            shortage_hospitals = shortage_hospitals[
                shortage_hospitals['hospital_id'].isin(shortage_hospital_ids)
            ].copy()
        
        shortage_count = len(shortage_hospitals)
        print(f"[Optimize Allocation] Found {shortage_count} hospitals with medium/high/critical shortage risk (after filtering)")
        
        # Calculate quantity_needed from shortage detection results
        # quantity_needed = predicted_demand_7d - current_stock (but ensure positive)
        if not shortage_hospitals.empty:
            # Get current inventory to calculate quantity needed
            current_inventory = self.data_loader.get_current_inventory()
            resource_inventory = current_inventory[
                current_inventory['resource_type'] == resource_type
            ]
            
            # Merge to get current stock
            shortage_hospitals = shortage_hospitals.merge(
                resource_inventory[['hospital_id', 'quantity', 'available_quantity']],
                on='hospital_id',
                how='left'
            )
            
            # Calculate quantity needed based on days of supply and predicted demand
            # If we have predicted_demand_7d from shortage detection, use it
            if 'predicted_demand_7d' in shortage_hospitals.columns:
                # quantity_needed = demand - current_stock, but at least days_of_supply suggests need
                shortage_hospitals['quantity_needed'] = (
                    shortage_hospitals.get('predicted_demand_7d', 0) - 
                    shortage_hospitals.get('current_stock', shortage_hospitals.get('quantity', 0))
                ).clip(lower=1)  # At least need 1 unit
            elif 'days_of_supply' in shortage_hospitals.columns:
                # Estimate from days_of_supply: if days < 7, need enough for 7 days
                shortage_hospitals['quantity_needed'] = shortage_hospitals.apply(
                    lambda row: max(1, int(row.get('current_stock', row.get('quantity', 0)) * 
                                          (7 / max(row.get('days_of_supply', 7), 0.1)) - 
                                          row.get('current_stock', row.get('quantity', 0)))),
                    axis=1
                )
            else:
                # Fallback: assume need 50% more than current stock for critical, 20% for high
                shortage_hospitals['quantity_needed'] = shortage_hospitals.apply(
                    lambda row: max(1, int(row.get('current_stock', row.get('quantity', 0)) * 
                                          (1.5 if row['risk_level'] == 'critical' else 1.2) - 
                                          row.get('current_stock', row.get('quantity', 0)))),
                    axis=1
                )
            
            # Ensure we have required columns
            required_cols = ['hospital_id', 'quantity_needed', 'risk_level']
            missing_cols = [col for col in required_cols if col not in shortage_hospitals.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in shortage_hospitals: {missing_cols}")
        
        # Get hospital info first (with limit if specified)
        hospitals = self.data_loader.get_hospitals(limit=hospital_limit)
        limited_hospital_ids = set(hospitals['hospital_id'].tolist()) if not hospitals.empty else set()
        print(f"[Optimize Allocation] Processing {len(limited_hospital_ids)} hospitals (limit: {hospital_limit})")
        
        # Find hospitals with surplus (filtered by hospital limit)
        current_inventory = self.data_loader.get_current_inventory()
        resource_inventory = current_inventory[
            current_inventory['resource_type'] == resource_type
        ]
        
        # Filter to only hospitals within the limit
        if limited_hospital_ids:
            resource_inventory = resource_inventory[
                resource_inventory['hospital_id'].isin(limited_hospital_ids)
            ].copy()
            print(f"[Optimize Allocation] Filtered inventory to {len(resource_inventory)} records for limited hospitals")
        
        # Surplus = available quantity > 1.5x critical threshold (relaxed from 2x)
        resource_types = self.data_loader.get_resource_types()
        threshold = resource_types[
            resource_types['name'] == resource_type
        ]['critical_threshold'].values[0]
        
        # Relaxed threshold: 1.5x instead of 2x to allow more transfers
        surplus_threshold = threshold * 1.5
        surplus_hospitals = resource_inventory[
            resource_inventory['available_quantity'] > surplus_threshold
        ].copy()
        
        surplus_count = len(surplus_hospitals)
        print(f"[Optimize Allocation] Found {surplus_count} hospitals with surplus (threshold: {surplus_threshold:.2f})")
        
        # Ensure surplus_hospitals has required columns
        if not surplus_hospitals.empty:
            required_surplus_cols = ['hospital_id', 'available_quantity']
            missing_surplus_cols = [col for col in required_surplus_cols if col not in surplus_hospitals.columns]
            if missing_surplus_cols:
                raise ValueError(f"Missing required columns in surplus_hospitals: {missing_surplus_cols}")
        
        # Optimize (hospitals already fetched above)
        result = self.optimizer.optimize_allocation(
            shortage_hospitals=shortage_hospitals,
            surplus_hospitals=surplus_hospitals,
            hospital_info=hospitals,
            resource_type=resource_type,
            objective_weights=objective_weights
        )
        
        # Add diagnostic information to result
        result['shortage_count'] = shortage_count
        result['surplus_count'] = surplus_count
        
        # If no feasible transfers, add detailed diagnostics
        if result.get('status') == 'no_feasible_transfers' or result.get('status') == 'no_feasible_transfers':
            diagnostics = {
                'shortage_count': shortage_count,
                'surplus_count': surplus_count,
                'surplus_threshold': float(surplus_threshold),
                'critical_threshold': float(threshold),
                'message': result.get('message', 'No surplus hospitals within range of shortage hospitals')
            }
            
            # Calculate distance distribution if we have hospital info
            if not hospitals.empty and not shortage_hospitals.empty and not surplus_hospitals.empty:
                try:
                    from ml_core.models.optimizer import ResourceOptimizer
                    temp_optimizer = ResourceOptimizer()
                    distances = []
                    distance_issues = 0
                    
                    for _, shortage in shortage_hospitals.iterrows():
                        shortage_id = shortage['hospital_id']
                        shortage_info = hospitals[hospitals['hospital_id'] == shortage_id]
                        if shortage_info.empty:
                            continue
                        shortage_info = shortage_info.iloc[0]
                        
                        for _, surplus in surplus_hospitals.iterrows():
                            surplus_id = surplus['hospital_id']
                            surplus_info = hospitals[hospitals['hospital_id'] == surplus_id]
                            if surplus_info.empty:
                                continue
                            surplus_info = surplus_info.iloc[0]
                            
                            distance = temp_optimizer.calculate_distance(
                                shortage_info.to_dict(),
                                surplus_info.to_dict()
                            )
                            distances.append(distance)
                            if distance > temp_optimizer.config['max_transfer_distance_km']:
                                distance_issues += 1
                    
                    if distances:
                        diagnostics['distance_stats'] = {
                            'min': float(min(distances)),
                            'max': float(max(distances)),
                            'mean': float(sum(distances) / len(distances)),
                            'over_limit_count': distance_issues,
                            'max_allowed': float(temp_optimizer.config['max_transfer_distance_km'])
                        }
                except Exception as e:
                    print(f"[Optimize Allocation] Warning: Could not calculate distance diagnostics: {e}")
            
            result['diagnostics'] = diagnostics
        
        return result
    
    def generate_allocation_strategies(
        self,
        resource_type: str,
        n_strategies: int = 3,
        hospital_limit: int = None
    ) -> List[Dict]:
        """
        Generate multiple allocation strategies
        
        Returns:
            List of ranked strategies
        """
        # Detect shortages
        all_shortages = self.detect_shortages(resource_type, hospital_limit=hospital_limit)
        # Include medium, high, and critical risk levels for strategy generation
        shortage_hospitals = all_shortages[
            all_shortages['risk_level'].isin(['medium', 'high', 'critical'])
        ].copy()
        print(f"[Generate Strategies] Found {len(all_shortages)} total shortage predictions")
        print(f"[Generate Strategies] Risk level breakdown: {all_shortages['risk_level'].value_counts().to_dict()}")
        
        # Calculate quantity_needed (same logic as optimize_allocation)
        if not shortage_hospitals.empty:
            current_inventory = self.data_loader.get_current_inventory()
            resource_inventory = current_inventory[
                current_inventory['resource_type'] == resource_type
            ]
            
            # Merge to get current stock
            shortage_hospitals = shortage_hospitals.merge(
                resource_inventory[['hospital_id', 'quantity', 'available_quantity']],
                on='hospital_id',
                how='left'
            )
            
            # Calculate quantity needed
            if 'predicted_demand_7d' in shortage_hospitals.columns:
                shortage_hospitals['quantity_needed'] = (
                    shortage_hospitals.get('predicted_demand_7d', 0) - 
                    shortage_hospitals.get('current_stock', shortage_hospitals.get('quantity', 0))
                ).clip(lower=1)
            elif 'days_of_supply' in shortage_hospitals.columns:
                shortage_hospitals['quantity_needed'] = shortage_hospitals.apply(
                    lambda row: max(1, int(row.get('current_stock', row.get('quantity', 0)) * 
                                          (7 / max(row.get('days_of_supply', 7), 0.1)) - 
                                          row.get('current_stock', row.get('quantity', 0)))),
                    axis=1
                )
            else:
                shortage_hospitals['quantity_needed'] = shortage_hospitals.apply(
                    lambda row: max(1, int(row.get('current_stock', row.get('quantity', 0)) * 
                                          (1.5 if row['risk_level'] == 'critical' else 1.2) - 
                                          row.get('current_stock', row.get('quantity', 0)))),
                    axis=1
                )
        
        # Get hospital info first (with limit if specified)
        hospitals = self.data_loader.get_hospitals(limit=hospital_limit)
        limited_hospital_ids = set(hospitals['hospital_id'].tolist()) if not hospitals.empty else set()
        print(f"[Generate Strategies] Processing {len(limited_hospital_ids)} hospitals (limit: {hospital_limit})")
        
        # Find surplus (filtered by hospital limit)
        current_inventory = self.data_loader.get_current_inventory()
        resource_inventory = current_inventory[
            current_inventory['resource_type'] == resource_type
        ]
        
        # Filter to only hospitals within the limit
        if limited_hospital_ids:
            resource_inventory = resource_inventory[
                resource_inventory['hospital_id'].isin(limited_hospital_ids)
            ].copy()
        
        resource_types = self.data_loader.get_resource_types()
        threshold = resource_types[
            resource_types['name'] == resource_type
        ]['critical_threshold'].values[0]
        
        # Relaxed threshold: 1.5x instead of 2x to allow more transfers
        surplus_threshold = threshold * 1.5
        surplus_hospitals = resource_inventory[
            resource_inventory['available_quantity'] > surplus_threshold
        ].copy()
        
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
        Update user preferences from interaction (with hybrid system)

        Args:
            user_id: User identifier
            interaction: Dict with interaction details including:
                - selected_recommendation_index: Index of chosen recommendation
                - recommendations: List of recommendations shown
                - timestamp: ISO timestamp
        """
        # Use enhanced method that stores in vector DB + updates RF model
        self.preference_learner.update_from_interaction_enhanced(user_id, interaction)
    
    def score_recommendations(
        self,
        recommendations: List[Dict],
        user_id: str = None,
        past_interactions: List[Dict] = None
    ) -> List[Dict]:
        """
        Score and re-rank recommendations using hybrid system (RF + LLM + Vector)

        Args:
            recommendations: List of recommendation dicts with allocation strategies
            user_id: Optional user ID for personalization
            past_interactions: Optional list of past user interactions for profile building

        Returns:
            Recommendations ranked by preference score with explanations
            Each recommendation has:
                - preference_score: Combined hybrid score (0-1)
                - score_breakdown: Dict with RF/LLM/Vector scores and weights
                - llm_explanation: Natural language explanation (if LLM enabled)
        """
        if not user_id:
            # Without user_id, use basic RF scoring
            return self.preference_learner.score_recommendations(
                recommendations=recommendations
            )

        # Build enhanced user profile with LLM insights
        user_profile = None
        if past_interactions:
            user_profile = self.preference_learner.get_user_profile_enhanced(
                user_id,
                past_interactions
            )

        # Use hybrid scoring (40% RF + 30% LLM + 30% Vector)
        return self.preference_learner.score_recommendations_hybrid(
            user_id=user_id,
            recommendations=recommendations,
            user_profile=user_profile
        )
    
    def get_user_preferences(
        self,
        user_id: str,
        interactions: List[Dict]
    ) -> Dict:
        """
        Get comprehensive user preference profile with LLM insights

        Args:
            user_id: User identifier
            interactions: List of past user interactions

        Returns:
            Dict with:
                - preference_type: Detected preference type (cost-conscious, coverage-focused, etc.)
                - preference_weights: Learned weights for cost/speed/coverage
                - llm_insights: Deep analysis from Llama 3.3 70B (if enabled)
                - key_patterns: List of behavioral patterns
                - confidence: Confidence in preference assessment (0-1)
                - vector_db_count: Number of interactions in vector store
        """
        return self.preference_learner.get_user_profile_enhanced(user_id, interactions)
    
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