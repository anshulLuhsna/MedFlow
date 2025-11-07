"""
Unified ML Core Interface
Single entry point for all ML functionality used by agent frameworks
"""

import numpy as np
import pandas as pd
import time
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
        predict_start = time.time()
        prediction = forecaster.predict_for_hospital(
            hospital_id=hospital_id,
            historical_data=data,
            days_ahead=days_ahead
        )
        predict_elapsed = time.time() - predict_start
        if predict_elapsed > 1.0:
            print(f"[Predict Demand] Model inference took {predict_elapsed:.2f}s for {hospital_id}")
        
        return prediction
    
    def predict_demand_all_hospitals(
        self,
        resource_type: str,
        days_ahead: int = 14,
        hospital_limit: int = None,
        hospital_ids: Optional[List[str]] = None,
        regions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Predict demand for all hospitals
        
        Args:
            resource_type: Type of resource to predict
            days_ahead: Number of days to forecast
            hospital_limit: Optional limit on number of hospitals to process (max 100)
            hospital_ids: Optional list of hospital IDs to process (filters to only these hospitals)
            regions: Optional list of regions to filter by
        
        Returns:
            DataFrame with predictions for all hospitals
        """
        # If hospital_ids provided, use limit=100 to get all, then filter
        # If regions provided, filter by regions
        # Default to 5 for demo performance if hospital_limit not provided
        # IMPORTANT: Use hospital_limit if provided, otherwise default to 5
        effective_limit = hospital_limit if hospital_limit is not None else 5
        hospitals = self.data_loader.get_hospitals(
            limit=effective_limit if not hospital_ids else 100,
            regions=regions
        )
        
        # Filter by hospital_ids if provided
        if hospital_ids:
            hospitals = hospitals[hospitals['hospital_id'].isin(hospital_ids)].copy()
            if hospitals.empty:
                print(f"[Predict Demand] No hospitals found matching provided IDs: {hospital_ids}")
                return pd.DataFrame()
        
        # Check if hospitals DataFrame is empty
        if hospitals.empty:
            print(f"[Predict Demand] No hospitals found for resource type {resource_type}")
            return pd.DataFrame()
        
        print(f"[Predict Demand] Predicting demand for {len(hospitals)} hospitals (resource: {resource_type})")
        print(f"[Predict Demand] Hospital limit: {effective_limit}, Actual hospitals: {len(hospitals)}")
        
        predict_start = time.time()
        predictions = []
        successful = 0
        failed = 0
        
        for idx, (_, hospital) in enumerate(hospitals.iterrows()):
            if idx % 5 == 0 and idx > 0:
                elapsed = time.time() - predict_start
                print(f"[Predict Demand] Progress: {idx}/{len(hospitals)} hospitals processed in {elapsed:.2f}s")
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
        
        predict_elapsed = time.time() - predict_start
        print(f"[Predict Demand] Completed: {successful} successful, {failed} failed in {predict_elapsed:.2f}s")
        if successful > 0:
            avg_time = predict_elapsed / successful
            print(f"[Predict Demand] Average time per prediction: {avg_time:.2f}s")
        
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
        hospital_limit: int = None,
        hospital_ids: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        simulation_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect shortage risks across all hospitals
        
        Args:
            resource_type: Optional filter for specific resource type
        
        Returns:
            DataFrame with shortage predictions
        """
        try:
            shortage_start = time.time()
            print(f"[Shortage Detection] Starting shortage detection (resource_type={resource_type}, limit={hospital_limit})")
            print(f"[Shortage Detection] Filters: hospital_ids={hospital_ids}, regions={regions}")
            
            if simulation_date:
                print(f"[Shortage Detection] 🔕 SIMULATION DATE: {simulation_date} - Using historical data as 'today'")
            
            # Get current inventory
            print("[Shortage Detection] Fetching current inventory...")
            current_inventory = self.data_loader.get_current_inventory(as_of_date=simulation_date)
            if current_inventory.empty:
                print("[Shortage Detection] Warning: No current inventory data found")
                return pd.DataFrame()
            print(f"[Shortage Detection] Found {len(current_inventory)} inventory records")
            
            # Get demand predictions
            print("[Shortage Detection] Generating demand predictions...")
            if resource_type:
                print(f"[Shortage Detection] Predicting demand for resource type: {resource_type}")
                demand_preds = self.predict_demand_all_hospitals(resource_type, hospital_limit=hospital_limit, hospital_ids=hospital_ids, regions=regions)
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
                        preds = self.predict_demand_all_hospitals(rt, hospital_limit=hospital_limit, hospital_ids=hospital_ids, regions=regions)
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
            
            # Get hospital info - need more hospitals for regional feature calculations
            # Even if we limit processing, we need hospital info for regional context
            print("[Shortage Detection] Fetching hospital info...")
            # Get more hospitals for regional feature calculations, but respect hospital_limit if provided
            # Use max(hospital_limit*2, 20) for regional context, but don't exceed what's needed
            if hospital_limit:
                hospital_info_limit = max(hospital_limit * 2, 20)  # Get 2x for regional context, min 20
            else:
                hospital_info_limit = 50 if not regions else None  # Default to 50 for regional calculations
            hospitals = self.data_loader.get_hospitals(limit=hospital_info_limit, regions=regions)
            if hospitals.empty:
                print("[Shortage Detection] Warning: No hospital data found")
                return pd.DataFrame()
            print(f"[Shortage Detection] Found {len(hospitals)} hospitals")
            
            # Also get more inventory for regional calculations
            # Regional features need context from other hospitals in the region
            if regions or hospital_limit:
                # Get inventory for regional context (at least 50 hospitals)
                # Use simulation_date if provided for historical data
                regional_inventory = self.data_loader.get_current_inventory(as_of_date=simulation_date)
                # Filter by regions if provided
                if regions and 'region' in regional_inventory.columns:
                    regional_inventory = regional_inventory[regional_inventory['region'].isin(regions)]
                elif not regions:
                    # If filtering by hospital_limit, get inventory for those hospitals plus regional context
                    # Get inventory for all hospitals first, then filter
                    regional_inventory = self.data_loader.get_current_inventory(as_of_date=simulation_date)
            else:
                regional_inventory = current_inventory
            
            # Filter inventory to only hospitals we're processing
            # BUT keep regional_inventory unfiltered for regional feature calculations
            if hospital_ids:
                print(f"[Shortage Detection] Filtering current_inventory to {len(hospital_ids)} specified hospitals")
                current_inventory = current_inventory[current_inventory['hospital_id'].isin(hospital_ids)]
                print(f"[Shortage Detection] Filtered current_inventory: {len(current_inventory)} rows")
                print(f"[Shortage Detection] Keeping regional_inventory unfiltered ({len(regional_inventory)} rows) for regional feature calculations")
            
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
            
            # Pass RAW demand predictions to shortage detector (with 'day' column)
            # The feature engineering function needs daily data to calculate features
            # Use current_inventory (filtered) but regional_inventory (unfiltered) for regional context
            print("[Shortage Detection] Running shortage detection model with raw demand predictions...")
            results = self.shortage_detector.detect_shortages(
                current_inventory=current_inventory,  # Use filtered inventory for matching
                demand_predictions=demand_preds,  # Pass RAW demand predictions (with 'day' column)
                admissions_history=admissions,
                hospital_info=hospitals,
                regional_inventory=regional_inventory  # Pass unfiltered regional inventory for regional features
            )
            
            # Filter results to only include hospitals from the limited set
            if hospital_limit and not hospital_ids:
                # Get hospital IDs from limited inventory
                limited_hospital_ids = set(current_inventory['hospital_id'].unique())
                results = results[results['hospital_id'].isin(limited_hospital_ids)].copy()
                print(f"[Shortage Detection] Filtered results to {len(results)} hospitals from limited set")
            
            shortage_elapsed = time.time() - shortage_start
            print(f"[Shortage Detection] Completed in {shortage_elapsed:.2f}s. Found {len(results)} shortage predictions")
            if hospital_limit:
                print(f"[Shortage Detection] Processed {len(results)} hospitals (limit: {hospital_limit})")
            return results
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[Shortage Detection] Error: {e}")
            print(f"[Shortage Detection] Traceback:\n{error_trace}")
            raise
    
    def get_shortage_summary(self, shortages_df: pd.DataFrame = None, hospital_limit: int = None) -> Dict:
        """
        Get summary of current shortage situation
        
        Args:
            shortages_df: Optional pre-computed shortages DataFrame (avoids re-computation)
            hospital_limit: Optional limit for hospitals if shortages_df is not provided
        
        Returns:
            Dict with aggregated shortage statistics
        """
        if shortages_df is None:
            # Only compute if not provided (with limit for demo)
            shortages = self.detect_shortages(hospital_limit=hospital_limit)
        else:
            shortages = shortages_df
        
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
        hospital_limit: int = None,
        simulation_date: Optional[str] = None
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
        
        # Log simulation date if provided
        if simulation_date:
            print(f"[Optimize Allocation] ✅ USING SIMULATION DATE: {simulation_date}")
        
        # Detect shortages if not provided
        if shortage_hospital_ids is None:
            all_shortages = self.detect_shortages(resource_type, hospital_limit=hospital_limit, simulation_date=simulation_date)
            # Include medium, high, and critical risk levels (not just high/critical)
            # This allows optimization to help hospitals before they reach critical status
            shortage_hospitals = all_shortages[
                all_shortages['risk_level'].isin(['medium', 'high', 'critical'])
            ].copy()
            print(f"[Optimize Allocation] Found {len(all_shortages)} total shortage predictions")
            print(f"[Optimize Allocation] Risk level breakdown: {all_shortages['risk_level'].value_counts().to_dict()}")
        else:
            shortage_hospitals = self.detect_shortages(resource_type, hospital_limit=hospital_limit, simulation_date=simulation_date)
            shortage_hospitals = shortage_hospitals[
                shortage_hospitals['hospital_id'].isin(shortage_hospital_ids)
            ].copy()
        
        shortage_count = len(shortage_hospitals)
        print(f"[Optimize Allocation] Found {shortage_count} hospitals with medium/high/critical shortage risk (after filtering)")
        
        # Calculate quantity_needed - shortage detection already has current_stock from feature engineering
        if not shortage_hospitals.empty:
            # Ensure we have current_stock column (should exist from shortage detection)
            if 'current_stock' not in shortage_hospitals.columns and 'stock_level' in shortage_hospitals.columns:
                shortage_hospitals['current_stock'] = shortage_hospitals['stock_level']
            
            # Calculate quantity needed based on available columns
            if 'predicted_demand_7d' in shortage_hospitals.columns and 'current_stock' in shortage_hospitals.columns:
                shortage_hospitals['quantity_needed'] = (
                    shortage_hospitals['predicted_demand_7d'] - 
                    shortage_hospitals['current_stock']
                ).clip(lower=1)
            elif 'days_of_supply' in shortage_hospitals.columns and 'current_stock' in shortage_hospitals.columns:
                shortage_hospitals['quantity_needed'] = shortage_hospitals.apply(
                    lambda row: max(1, int(row['current_stock'] * 
                                          (7 / max(row.get('days_of_supply', 7), 0.1)) - 
                                          row['current_stock'])),
                    axis=1
                )
            else:
                # Fallback: estimate based on risk level
                shortage_hospitals['quantity_needed'] = shortage_hospitals.apply(
                    lambda row: max(1, int(row.get('current_stock', row.get('stock_level', 10)) * 
                                          (0.5 if row['risk_level'] == 'critical' else 0.3))),
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
        print(f"[Optimize Allocation] Processing {len(limited_hospital_ids)} hospitals for shortages (limit: {hospital_limit})")
        
        # Find hospitals with surplus - IMPORTANT: Use ALL hospitals, not just limited ones
        # This allows transfers from hospitals outside the limited set
        print(f"[Optimize Allocation] Searching for surplus hospitals across ALL hospitals (not limited)")
        current_inventory_all = self.data_loader.get_current_inventory(as_of_date=simulation_date)
        resource_inventory_all = current_inventory_all[
            current_inventory_all['resource_type'] == resource_type
        ]
        
        # Find hospitals with surplus (filtered by hospital limit)
        current_inventory = self.data_loader.get_current_inventory(as_of_date=simulation_date)
        resource_inventory = current_inventory[
            current_inventory['resource_type'] == resource_type
        ]
        
        # Filter to only hospitals within the limit (for shortage hospitals only)
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
        
        # Use ALL hospitals for surplus search (not just limited ones)
        surplus_hospitals = resource_inventory_all[
            resource_inventory_all['available_quantity'] > surplus_threshold
        ].copy()
        
        # Also need hospital info for ALL hospitals (for distance calculations)
        hospitals_all = self.data_loader.get_hospitals(limit=None)  # Get all hospitals
        
        surplus_count = len(surplus_hospitals)
        print(f"[Optimize Allocation] Found {surplus_count} hospitals with surplus (from ALL hospitals, threshold: {surplus_threshold:.2f})")
        
        # Ensure surplus_hospitals has required columns
        if not surplus_hospitals.empty:
            required_surplus_cols = ['hospital_id', 'available_quantity']
            missing_surplus_cols = [col for col in required_surplus_cols if col not in surplus_hospitals.columns]
            if missing_surplus_cols:
                raise ValueError(f"Missing required columns in surplus_hospitals: {missing_surplus_cols}")
        
        # Optimize (use all hospitals for distance calculations)
        result = self.optimizer.optimize_allocation(
            shortage_hospitals=shortage_hospitals,
            surplus_hospitals=surplus_hospitals,
            hospital_info=hospitals_all,  # Use all hospitals for distance calculations
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
        hospital_limit: int = None,
        hospital_ids: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        simulation_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate multiple allocation strategies
        
        Returns:
            List of ranked strategies
        """
        strategy_start = time.time()
        print(f"[Generate Strategies] Starting strategy generation (n_strategies={n_strategies}, limit={hospital_limit})")
        
        # Log simulation date if provided
        if simulation_date:
            print(f"[Generate Strategies] ✅ USING SIMULATION DATE: {simulation_date}")
        
        # Detect shortages (with hospital_ids or regions filter if provided)
        all_shortages = self.detect_shortages(resource_type, hospital_limit=hospital_limit, hospital_ids=hospital_ids, regions=regions, simulation_date=simulation_date)
        # Include medium, high, and critical risk levels for strategy generation
        shortage_hospitals = all_shortages[
            all_shortages['risk_level'].isin(['medium', 'high', 'critical'])
        ].copy()
        print(f"[Generate Strategies] Found {len(all_shortages)} total shortage predictions")
        print(f"[Generate Strategies] Risk level breakdown: {all_shortages['risk_level'].value_counts().to_dict()}")
        
        # Calculate quantity_needed - shortage detection already has current_stock from feature engineering
        if not shortage_hospitals.empty:
            # Ensure we have current_stock column (should exist from shortage detection)
            if 'current_stock' not in shortage_hospitals.columns and 'stock_level' in shortage_hospitals.columns:
                shortage_hospitals['current_stock'] = shortage_hospitals['stock_level']
            
            # Calculate quantity needed based on available columns
            if 'predicted_demand_7d' in shortage_hospitals.columns and 'current_stock' in shortage_hospitals.columns:
                shortage_hospitals['quantity_needed'] = (
                    shortage_hospitals['predicted_demand_7d'] - 
                    shortage_hospitals['current_stock']
                ).clip(lower=1)
            elif 'days_of_supply' in shortage_hospitals.columns and 'current_stock' in shortage_hospitals.columns:
                shortage_hospitals['quantity_needed'] = shortage_hospitals.apply(
                    lambda row: max(1, int(row['current_stock'] * 
                                          (7 / max(row.get('days_of_supply', 7), 0.1)) - 
                                          row['current_stock'])),
                    axis=1
                )
            else:
                # Fallback: estimate based on risk level
                shortage_hospitals['quantity_needed'] = shortage_hospitals.apply(
                    lambda row: max(1, int(row.get('current_stock', row.get('stock_level', 10)) * 
                                          (0.5 if row['risk_level'] == 'critical' else 0.3))),
                    axis=1
                )
        
        # Get hospital info first (with limit if specified, or filter by hospital_ids/regions)
        if hospital_ids:
            # If hospital_ids provided, get all then filter
            hospitals = self.data_loader.get_hospitals(limit=100, regions=regions)
            hospitals = hospitals[hospitals['hospital_id'].isin(hospital_ids)].copy()
            limited_hospital_ids = set(hospital_ids)  # Use provided IDs
            print(f"[Generate Strategies] Processing {len(limited_hospital_ids)} specified hospitals")
        elif regions:
            # If regions provided, filter by regions
            # Use hospital_limit if provided, otherwise default to 5 for demo performance
            hospitals = self.data_loader.get_hospitals(limit=hospital_limit or 5, regions=regions)
            limited_hospital_ids = set(hospitals['hospital_id'].tolist()) if not hospitals.empty else set()
            print(f"[Generate Strategies] Processing {len(limited_hospital_ids)} hospitals from regions: {regions}")
        else:
            hospitals = self.data_loader.get_hospitals(limit=hospital_limit)
            limited_hospital_ids = set(hospitals['hospital_id'].tolist()) if not hospitals.empty else set()
            print(f"[Generate Strategies] Processing {len(limited_hospital_ids)} hospitals (limit: {hospital_limit})")
        
        # Find surplus - IMPORTANT: Use ALL hospitals, not just limited ones
        # This allows transfers from hospitals outside the limited set
        print(f"[Generate Strategies] Searching for surplus hospitals across ALL hospitals (not limited)")
        current_inventory_all = self.data_loader.get_current_inventory(as_of_date=simulation_date)
        resource_inventory_all = current_inventory_all[
            current_inventory_all['resource_type'] == resource_type
        ]
        
        resource_types = self.data_loader.get_resource_types()
        threshold = resource_types[
            resource_types['name'] == resource_type
        ]['critical_threshold'].values[0]
        
        # Relaxed threshold: 1.5x instead of 2x to allow more transfers
        surplus_threshold = threshold * 1.5
        
        # Use ALL hospitals for surplus search (not just limited ones)
        surplus_hospitals = resource_inventory_all[
            resource_inventory_all['available_quantity'] > surplus_threshold
        ].copy()
        
        # Also need hospital info for ALL hospitals (for distance calculations)
        hospitals_all = self.data_loader.get_hospitals(limit=None)  # Get all hospitals
        
        print(f"[Generate Strategies] Found {len(surplus_hospitals)} hospitals with surplus (from ALL hospitals)")
        
        # Generate strategies
        print(f"[Generate Strategies] Generating {n_strategies} strategies...")
        opt_start = time.time()
        strategies = self.optimizer.generate_multiple_strategies(
            shortage_hospitals=shortage_hospitals,
            surplus_hospitals=surplus_hospitals,
            hospital_info=hospitals_all,  # Use all hospitals for distance calculations
            resource_type=resource_type,
            n_strategies=n_strategies
        )
        opt_elapsed = time.time() - opt_start
        strategy_elapsed = time.time() - strategy_start
        print(f"[Generate Strategies] Optimization completed in {opt_elapsed:.2f}s")
        print(f"[Generate Strategies] Total strategy generation took {strategy_elapsed:.2f}s")
        
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