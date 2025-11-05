"""
Data loading utilities for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from supabase import Client
import os
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    """Load data from Supabase for ML training and inference"""
    
    def __init__(self, supabase_client: Client = None):
        if supabase_client is None:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")
            self.client = create_client(url, key)
        else:
            self.client = supabase_client
    
    def _fetch_all_pages(self, query, table_name: str, verbose: bool = False):
        """
        Fetch all pages from Supabase query (handles pagination automatically)
        
        Args:
            query: Supabase query object
            table_name: Name of table being queried (for logging)
            verbose: Whether to print progress
        
        Returns:
            List of all records from all pages
        """
        all_data = []
        page_size = 1000
        offset = 0
        
        if verbose:
            print(f"  Fetching all records from {table_name} (paginated)...")
        
        while True:
            try:
                # Fetch one page at a time
                page_query = query.range(offset, offset + page_size - 1)
                response = page_query.execute()
                page_data = response.data
                
                if not page_data:
                    break
                
                all_data.extend(page_data)
                
                if verbose and len(all_data) % 10000 == 0:
                    print(f"    Fetched {len(all_data):,} records...")
                
                # If we got fewer records than page_size, we're on the last page
                if len(page_data) < page_size:
                    break
                
                offset += page_size
                
            except Exception as e:
                if verbose:
                    print(f"    Error fetching page at offset {offset}: {e}")
                # Try to continue, but if this is first page, we might not have data
                if offset == 0:
                    # Fallback to single query if pagination fails
                    if verbose:
                        print(f"    Falling back to single query...")
                    response = query.execute()
                    return response.data
                break
        
        if verbose:
            print(f"  ✓ Fetched {len(all_data):,} total records from {table_name}")
        
        return all_data
    
    def get_hospitals(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch hospitals with optional limit
        
        Args:
            limit: Optional limit on number of hospitals to fetch (max 100)
        
        Returns:
            DataFrame with hospital data
        """
        query = self.client.table("hospitals").select("*")
        
        # Apply limit if provided
        if limit is not None:
            if limit < 1 or limit > 100:
                raise ValueError("limit must be between 1 and 100")
            query = query.limit(limit)
            response = query.execute()
            df = pd.DataFrame(response.data)
        else:
            # Fetch all hospitals (with pagination)
            all_data = self._fetch_all_pages(query, "hospitals", verbose=False)
            df = pd.DataFrame(all_data)
        
        # Rename 'id' to 'hospital_id' for consistency with other tables
        if not df.empty and 'id' in df.columns:
            df = df.rename(columns={'id': 'hospital_id'})
        
        return df
    
    def get_resource_types(self) -> pd.DataFrame:
        """Fetch resource types with mapping"""
        response = self.client.table("resource_types").select("*").execute()
        return pd.DataFrame(response.data)
    
    def get_inventory_history(
        self,
        start_date: str = None,
        end_date: str = None,
        hospital_ids: List[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Fetch inventory history with optional filters (with pagination)
        
        Args:
            start_date: ISO format date string
            end_date: ISO format date string
            hospital_ids: List of hospital UUIDs
            verbose: Whether to print pagination progress
        """
        query = self.client.table("inventory_history").select("*")
        
        if start_date:
            query = query.gte("date", start_date)
        if end_date:
            query = query.lte("date", end_date)
        if hospital_ids:
            query = query.in_("hospital_id", hospital_ids)
        
        # Use pagination to fetch all records
        all_data = self._fetch_all_pages(query, "inventory_history", verbose=verbose)
        
        # Handle empty result
        if not all_data:
            # Return empty DataFrame with expected columns structure
            # This allows the caller to handle "no data" case gracefully
            df = pd.DataFrame(columns=['date', 'hospital_id', 'resource_type_id', 'quantity', 
                                      'consumption', 'resupply'])
            df['date'] = pd.to_datetime([])
            return df
        
        df = pd.DataFrame(all_data)
        
        # Ensure date column exists and is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif not df.empty:
            # If DataFrame has data but no 'date' column, check for alternative names
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                # Rename the first date-like column to 'date'
                df = df.rename(columns={date_cols[0]: 'date'})
                df['date'] = pd.to_datetime(df['date'])
            else:
                # Create a date column from index or use current date
                if 'created_at' in df.columns:
                    df['date'] = pd.to_datetime(df['created_at'])
                elif 'updated_at' in df.columns:
                    df['date'] = pd.to_datetime(df['updated_at'])
                else:
                    # If no date column at all, we can't proceed
                    raise ValueError(
                        f"inventory_history query returned data without 'date' column. "
                        f"Available columns: {df.columns.tolist()}"
                    )
        
        return df
    
    def get_admissions_history(
        self,
        start_date: str = None,
        end_date: str = None,
        hospital_ids: List[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Fetch patient admissions history (with pagination)
        
        Args:
            start_date: ISO format date string
            end_date: ISO format date string
            hospital_ids: List of hospital UUIDs
            verbose: Whether to print pagination progress
        """
        query = self.client.table("patient_admissions").select("*")
        
        if start_date:
            query = query.gte("admission_date", start_date)
        if end_date:
            query = query.lte("admission_date", end_date)
        if hospital_ids:
            query = query.in_("hospital_id", hospital_ids)
        
        # Use pagination to fetch all records
        all_data = self._fetch_all_pages(query, "patient_admissions", verbose=verbose)
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            df['admission_date'] = pd.to_datetime(df['admission_date'])
        
        return df
    
    def get_current_inventory(self) -> pd.DataFrame:
        """Fetch current inventory state"""
        response = self.client.table("resource_inventory").select(
            "*, hospitals(name, region, capacity_beds), resource_types(name, critical_threshold)"
        ).execute()
        df = pd.DataFrame(response.data)
        
        # Flatten nested data from Supabase joins
        if not df.empty:
            # Extract resource_types.name -> resource_type column
            if 'resource_types' in df.columns:
                resource_types_dict = df['resource_types'].apply(lambda x: x if isinstance(x, dict) else {})
                df['resource_type'] = resource_types_dict.apply(lambda x: x.get('name', ''))
                # Extract critical_threshold
                df['critical_threshold'] = resource_types_dict.apply(lambda x: x.get('critical_threshold', 0))
            
            # Extract hospitals data
            if 'hospitals' in df.columns:
                hospitals_dict = df['hospitals'].apply(lambda x: x if isinstance(x, dict) else {})
                df['hospital_name'] = hospitals_dict.apply(lambda x: x.get('name', ''))
                df['region'] = hospitals_dict.apply(lambda x: x.get('region', ''))
                df['capacity_beds'] = hospitals_dict.apply(lambda x: x.get('capacity_beds', 0))
                # If region column doesn't exist from flatten, use from hospitals
                if 'region' not in df.columns:
                    df['region'] = hospitals_dict.apply(lambda x: x.get('region', ''))
            
            # Calculate capacity: for beds use capacity_beds, for others use a default based on quantity
            # This is a heuristic - in production you might want resource-specific capacity storage
            if 'capacity_beds' in df.columns and 'resource_type' in df.columns:
                df['capacity'] = df.apply(
                    lambda row: row['capacity_beds'] if row['resource_type'] == 'beds' 
                    else max(row['quantity'] * 3, 100),  # Default: 3x current quantity or 100, whichever is larger
                    axis=1
                )
            elif 'capacity_beds' in df.columns:
                df['capacity'] = df['capacity_beds']
            else:
                # Fallback: use a default capacity
                df['capacity'] = 100
        
        return df
    
    def get_events(
        self,
        start_date: str = None,
        end_date: str = None,
        event_type: str = None,
        severity: str = None,
        region: str = None,
        active_only: bool = False,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fetch events (outbreaks, disruptions) with optional filters
        
        Args:
            start_date: ISO date string - filter events starting on/after this date
            end_date: ISO date string - filter events ending on/before this date
            event_type: Filter by type (outbreak, supply_disruption)
            severity: Filter by severity (low, medium, high, critical)
            region: Filter by affected region
            active_only: Only return events where current date is between start_date and end_date
            limit: Limit number of results (1-100)
        
        Returns:
            DataFrame with event data
        """
        from datetime import datetime
        
        query = self.client.table("events").select("*")
        
        if start_date:
            query = query.gte("start_date", start_date)
        if end_date:
            query = query.lte("end_date", end_date)
        if event_type:
            query = query.eq("event_type", event_type)
        if severity:
            query = query.eq("severity", severity)
        if region:
            # Note: affected_region is stored as comma-separated string, so we use ilike for partial match
            query = query.ilike("affected_region", f"%{region}%")
        
        # Active only: current date must be between start_date and end_date
        # Note: We need to filter after fetching since Supabase doesn't support complex date comparisons
        # For now, we'll fetch all and filter in Python
        if active_only:
            # We'll filter after fetching since we need to check if today is between start_date and end_date
            pass  # Will filter after query execution
        
        # Apply limit if specified
        if limit is not None:
            if limit < 1 or limit > 100:
                raise ValueError("limit must be between 1 and 100")
            query = query.limit(limit)
        
        response = query.execute()
        df = pd.DataFrame(response.data)
        
        if not df.empty:
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            
            # Filter for active events: current date must be between start_date and end_date
            if active_only:
                today = pd.Timestamp.now().date()
                df = df[
                    (df['start_date'].dt.date <= today) & 
                    ((df['end_date'].isna()) | (df['end_date'].dt.date >= today))
                ].copy()
        
        return df
    
    def prepare_training_data(
        self,
        resource_type: str,
        sequence_length: int = 30,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare data for demand forecasting model training
        
        Returns:
            X: Input sequences (samples, sequence_length, features)
            y: Target values (samples, forecast_horizon)
            metadata: DataFrame with hospital and date info
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Preparing training data for: {resource_type}")
            print(f"{'='*60}")
        
        # Get resource type ID
        resource_types = self.get_resource_types()
        resource_id = resource_types[resource_types['name'] == resource_type]['id'].values[0]
        
        if verbose:
            print(f"Resource type ID: {resource_id}")
        
        # Fetch inventory history (with pagination, verbose logging)
        inventory = self.get_inventory_history(verbose=verbose)
        if verbose:
            print(f"Fetched inventory: {len(inventory):,} records")
            print(f"Unique hospitals in inventory: {inventory['hospital_id'].nunique()}")
            print(f"Date range: {inventory['date'].min()} to {inventory['date'].max()}")
        
        inventory = inventory[inventory['resource_type_id'] == resource_id]
        if verbose:
            print(f"After filtering for {resource_type}: {len(inventory):,} records")
            print(f"Unique hospitals: {inventory['hospital_id'].nunique()}")
        
        # Fetch admissions (correlated feature) with pagination
        admissions = self.get_admissions_history(verbose=verbose)
        if verbose:
            print(f"Fetched admissions: {len(admissions):,} records")
            print(f"Unique hospitals in admissions: {admissions['hospital_id'].nunique()}")
            print(f"Date range: {admissions['admission_date'].min()} to {admissions['admission_date'].max()}")
        
        # Ensure date types are consistent for merge
        inventory['date'] = pd.to_datetime(inventory['date']).dt.date
        admissions['admission_date'] = pd.to_datetime(admissions['admission_date']).dt.date
        
        # Ensure hospital_id types match (convert both to string for safety)
        inventory['hospital_id'] = inventory['hospital_id'].astype(str)
        admissions['hospital_id'] = admissions['hospital_id'].astype(str)
        
        # Rename admission_date to date for merge
        admissions_for_merge = admissions.rename(columns={'admission_date': 'date'})
        
        # Merge data
        data = inventory.merge(
            admissions_for_merge[['hospital_id', 'date', 'total_admissions', 'icu_admissions']],
            on=['hospital_id', 'date'],
            how='left'
        )
        
        if verbose:
            print(f"After merge: {len(data)} records")
            print(f"Unique hospitals after merge: {data['hospital_id'].nunique()}")
            print(f"Records with missing admissions: {data['total_admissions'].isna().sum()}")
        
        # Fill missing admissions with 0
        data['total_admissions'] = data['total_admissions'].fillna(0).astype(int)
        data['icu_admissions'] = data['icu_admissions'].fillna(0).astype(int)
        
        # Ensure date is datetime for sorting
        data['date'] = pd.to_datetime(data['date'])
        
        # Sort by hospital and date
        data = data.sort_values(['hospital_id', 'date'])
        
        # Add engineered features for better directional learning
        from .feature_engineering import calculate_rolling_stats, calculate_trend
        
        # Add rolling statistics and trends
        enhanced_data = []
        hospitals = data['hospital_id'].unique()
        
        for hospital_id in hospitals:
            hospital_data = data[data['hospital_id'] == hospital_id].copy().sort_values('date')
            
            # Calculate rolling means for quantity (7-day and 14-day)
            hospital_data['quantity_ma_7d'] = hospital_data['quantity'].rolling(window=7, min_periods=1).mean()
            hospital_data['quantity_ma_14d'] = hospital_data['quantity'].rolling(window=14, min_periods=1).mean()
            
            # Calculate trend (slope over last 14 days)
            hospital_data['quantity_trend'] = calculate_trend(hospital_data['quantity'], window=14)
            
            # Calculate consumption trend
            hospital_data['consumption_trend'] = calculate_trend(hospital_data['consumption'], window=14)
            
            # Add rate of change features
            hospital_data['quantity_change'] = hospital_data['quantity'].diff().fillna(0)
            hospital_data['consumption_change'] = hospital_data['consumption'].diff().fillna(0)

            # Add relative features (normalized by admissions)
            hospital_data['quantity_per_admission'] = hospital_data['quantity'] / (hospital_data['total_admissions'] + 1)
            hospital_data['consumption_rate'] = hospital_data['consumption'] / (hospital_data['total_admissions'] + 1)

            # NEW: Momentum features (2nd derivative - acceleration)
            # These help the model understand if the rate of change is accelerating/decelerating
            hospital_data['quantity_momentum'] = hospital_data['quantity_change'].diff().fillna(0)
            hospital_data['consumption_momentum'] = hospital_data['consumption_change'].diff().fillna(0)

            # NEW: Percentage changes (more robust to scale differences)
            # Percentage changes are better for detecting relative trends
            hospital_data['quantity_pct_change'] = hospital_data['quantity'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            hospital_data['consumption_pct_change'] = hospital_data['consumption'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

            # NEW: Trend direction indicator (-1, 0, or 1)
            # Explicit signal about whether quantity is trending up or down
            hospital_data['trend_direction'] = np.sign(hospital_data['quantity_trend']).fillna(0)
            
            enhanced_data.append(hospital_data)
        
        # Combine all hospitals
        data = pd.concat(enhanced_data, ignore_index=True)
        
        # Create sequences for each hospital
        X_list, y_list, metadata_list = [], [], []
        
        hospitals = data['hospital_id'].unique()
        forecast_horizon = 14
        min_days_required = sequence_length + forecast_horizon
        
        if verbose:
            print(f"\nProcessing {len(hospitals)} hospitals...")
            print(f"Minimum days required per hospital: {min_days_required}")
        
        hospitals_processed = 0
        hospitals_skipped = 0
        total_sequences = 0
        
        for hospital_id in hospitals:
            hospital_data = data[data['hospital_id'] == hospital_id].copy()
            hospital_data = hospital_data.sort_values('date')
            
            # Check if hospital has enough data
            if len(hospital_data) < min_days_required:
                hospitals_skipped += 1
                if verbose and hospitals_skipped <= 5:  # Log first 5 skipped
                    print(f"  ⚠ Skipping hospital {hospital_id[:8]}... (only {len(hospital_data)} days, need {min_days_required})")
                continue
            
            # Extract features and target - now with enhanced features
            feature_cols = [
                # Base features (4)
                'quantity', 'consumption', 'resupply', 'total_admissions',
                # Trend features (4)
                'quantity_ma_7d', 'quantity_ma_14d', 'quantity_trend', 'consumption_trend',
                # Change features (2)
                'quantity_change', 'consumption_change',
                # Normalized features (2)
                'quantity_per_admission', 'consumption_rate',
                # NEW: Momentum features (2) - acceleration/deceleration
                'quantity_momentum', 'consumption_momentum',
                # NEW: Percentage change features (2) - relative trends
                'quantity_pct_change', 'consumption_pct_change',
                # NEW: Directional indicator (1) - explicit trend direction
                'trend_direction'
            ]
            # Total: 17 features (was 12)
            
            # Ensure all columns exist (fill missing with 0)
            for col in feature_cols:
                if col not in hospital_data.columns:
                    hospital_data[col] = 0
            
            features = hospital_data[feature_cols].fillna(0).astype(float).values
            
            # Create sequences
            hospital_sequences = 0
            for i in range(len(features) - sequence_length - forecast_horizon):
                X_list.append(features[i:i+sequence_length])
                # CHANGED: Predict consumption (index 1) instead of quantity (index 0)
                # Consumption is the actual demand, which is predictable from admissions
                # Quantity includes random resupply decisions, making it unpredictable
                y_list.append(features[i+sequence_length:i+sequence_length+forecast_horizon, 1])  # Predict consumption
                metadata_list.append({
                    'hospital_id': hospital_id,
                    'date': hospital_data.iloc[i+sequence_length]['date']
                })
                hospital_sequences += 1
            
            total_sequences += hospital_sequences
            hospitals_processed += 1
            
            if verbose and hospitals_processed <= 5:  # Log first 5 hospitals
                print(f"  ✓ Hospital {hospital_id[:8]}...: {hospital_sequences} sequences ({len(hospital_data)} days)")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Summary:")
            print(f"  Hospitals processed: {hospitals_processed}/{len(hospitals)}")
            print(f"  Hospitals skipped: {hospitals_skipped}")
            print(f"  Total sequences generated: {len(X_list)}")
            print(f"{'='*60}\n")
        
        if len(X_list) == 0:
            raise ValueError(f"No training sequences generated for {resource_type}. Check data completeness.")
        
        X = np.array(X_list)
        y = np.array(y_list)
        metadata = pd.DataFrame(metadata_list)
        
        return X, y, metadata


def create_time_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Add time-based features"""
    df = df.copy()
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # Cyclical encoding
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df