"""
Data processor module for EV charging prediction.
Handles feature engineering and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import Dataset, DataLoader
import datetime
import joblib
import os
from ..config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET, MAX_SEQUENCE_LENGTH

class EVDataProcessor:
    """
    Process raw EV data into features suitable for machine learning models.
    Extracts meaningful features from raw trip data.
    """
    
    def __init__(self, save_dir=None):
        """
        Initialize the data processor.
        
        Args:
            save_dir: Directory to save/load preprocessors
        """
        self.save_dir = save_dir
        
        # Create preprocessor pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
            ],
            remainder='drop'
        )
        
        self.fitted = False
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.preprocessor_path = os.path.join(save_dir, 'preprocessor.joblib')
            
            # Load preprocessor if it exists
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                self.fitted = True
                print(f"Loaded preprocessor from {self.preprocessor_path}")
    
    def process_data(self, data):
        """
        Process raw data and extract features.
        
        Args:
            data: DataFrame containing the raw data
            
        Returns:
            DataFrame with extracted features
        """
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Convert date and time strings to datetime objects
        processed_data['start_datetime'] = pd.to_datetime(
            processed_data['start_date'] + ' ' + processed_data['start_time'].str.split(' ').str[1], 
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        
        processed_data['end_datetime'] = pd.to_datetime(
            processed_data['end_date'] + ' ' + processed_data['end_time'].str.split(' ').str[1], 
            format='%Y-%m-%d %H:%M:%S', 
            errors='coerce'
        )
        
        # Extract time-related features
        processed_data['day_of_week'] = processed_data['start_datetime'].dt.day_name()
        processed_data['hour'] = processed_data['start_datetime'].dt.hour
        processed_data['month'] = processed_data['start_datetime'].dt.month
        processed_data['day_of_year'] = processed_data['start_datetime'].dt.dayofyear
        
        # Encode cyclical time features
        processed_data['hour_sin'] = np.sin(2 * np.pi * processed_data['hour'] / 24)
        processed_data['hour_cos'] = np.cos(2 * np.pi * processed_data['hour'] / 24)
        processed_data['day_sin'] = np.sin(2 * np.pi * processed_data['day_of_year'] / 365)
        processed_data['day_cos'] = np.cos(2 * np.pi * processed_data['day_of_year'] / 365)
        
        # Categorize time of day
        conditions = [
            (processed_data['hour'] >= 5) & (processed_data['hour'] < 12),
            (processed_data['hour'] >= 12) & (processed_data['hour'] < 17),
            (processed_data['hour'] >= 17) & (processed_data['hour'] < 22),
            (processed_data['hour'] >= 22) | (processed_data['hour'] < 5)
        ]
        time_categories = ['morning', 'afternoon', 'evening', 'night']
        processed_data['time_of_day'] = np.select(conditions, time_categories)
        
        # Calculate trip duration in hours
        processed_data['trip_duration_hours'] = (
            processed_data['end_datetime'] - processed_data['start_datetime']
        ).dt.total_seconds() / 3600
        
        # Calculate approximate trip distance in km (using Haversine formula)
        processed_data['trip_distance_km'] = self._calculate_distance(
            processed_data['start_location_latitude'], 
            processed_data['start_location_longitude'],
            processed_data['end_location_latitude'], 
            processed_data['end_location_longitude']
        )
        
        # Create target variable (whether to charge)
        # If the type is 'charge', the target is 1, otherwise 0
        processed_data['should_charge'] = (processed_data['type'] == 'charge').astype(int)
        
        # Calculate battery drain rate where applicable
        processed_data['battery_drain_per_km'] = processed_data.apply(
            lambda row: ((row['end_battery_level'] - 100) / row['trip_distance_km'] 
                        if row['type'] == 'charge' else 
                        row['end_battery_level'] / row['trip_distance_km']), 
            axis=1
        )
        
        # Add user history features (rolling window statistics)
        processed_data = self._add_history_features(processed_data)
        
        # Extract user-specific features
        processed_data = self._extract_user_features(processed_data)
        
        return processed_data
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth using the Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of point 1
            lat2, lon2: Latitude and longitude of point 2
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    def _add_history_features(self, data):
        """
        Add features based on user history using rolling windows.
        
        Args:
            data: Processed DataFrame
            
        Returns:
            DataFrame with added history features
        """
        # Sort by user and datetime for proper history tracking
        data = data.sort_values(['user_car_id', 'start_datetime'])
        
        # Group by user_car_id to calculate user-specific features
        # Calculate rolling averages for key metrics
        # We need to ensure that transform is applied on the original group, not a sub-selection
        # if the column for transform is created within the same chain.
        
        # Calculate rolling averages for key metrics directly on the original DataFrame
        # after grouping, to avoid issues with chained assignments on copies.
        
        # Create a new DataFrame for group-wise calculations to avoid SettingWithCopyWarning
        # and ensure operations are on the correct data.
        
        # Calculate rolling averages for key metrics
        for col in ['trip_distance_km', 'trip_duration_hours', 'end_battery_level']:
            data[f'avg_{col}_7d'] = data.groupby('user_car_id')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )

        # Calculate time since last charge correctly
        # Make a copy of the data to avoid potential issues with modifications
        data = data.copy()
        
        # Only process rows with valid start_datetime
        valid_datetime_mask = ~pd.isna(data['start_datetime'])
        
        # Initialize days_since_last_charge column with NaN
        data['days_since_last_charge'] = np.nan
        
        if valid_datetime_mask.any():
            # Mark charge events
            data['is_charge_event'] = (data['type'] == 'charge')
            
            # Create a temporary column for the start_datetime of charge events
            data.loc[data['is_charge_event'] & valid_datetime_mask, 'charge_event_datetime'] = data.loc[data['is_charge_event'] & valid_datetime_mask, 'start_datetime']
            
            # Forward fill the datetime of the last charge event for each user
            data['last_charge_datetime'] = data.groupby('user_car_id')['charge_event_datetime'].ffill()
            
            # Ensure 'start_datetime' and 'last_charge_datetime' are both timezone-naive or both timezone-aware
            datetime_cols_valid = valid_datetime_mask & ~pd.isna(data['last_charge_datetime'])
            
            if datetime_cols_valid.any():
                if pd.api.types.is_datetime64_any_dtype(data['start_datetime']) and data['start_datetime'].dt.tz is not None:
                    if pd.api.types.is_datetime64_any_dtype(data['last_charge_datetime']) and data['last_charge_datetime'].dt.tz is None:
                        data.loc[datetime_cols_valid, 'last_charge_datetime'] = data.loc[datetime_cols_valid, 'last_charge_datetime'].dt.tz_localize(data['start_datetime'].dt.tz)
                elif pd.api.types.is_datetime64_any_dtype(data['last_charge_datetime']) and data['last_charge_datetime'].dt.tz is not None:
                    if pd.api.types.is_datetime64_any_dtype(data['start_datetime']) and data['start_datetime'].dt.tz is None:
                        data.loc[datetime_cols_valid, 'start_datetime'] = data.loc[datetime_cols_valid, 'start_datetime'].dt.tz_localize(data['last_charge_datetime'].dt.tz)

                # Calculate days since last charge only where both datetimes are valid
                both_dates_valid = valid_datetime_mask & ~pd.isna(data['last_charge_datetime'])
                if both_dates_valid.any():
                    # Calculate the time difference and convert to days
                    data.loc[both_dates_valid, 'days_since_last_charge'] = (
                        data.loc[both_dates_valid, 'start_datetime'] - 
                        data.loc[both_dates_valid, 'last_charge_datetime']
                    ).dt.total_seconds() / (3600 * 24)

        # Clean up temporary columns
        data = data.drop(columns=['is_charge_event', 'charge_event_datetime', 'last_charge_datetime'], errors='ignore')
        
        # Calculate charge frequency (number of charge events in the last 30 days for each user)
        def count_charges_last_30d(group):
            # Return zeros if group is empty
            if group.empty:
                return pd.Series(0, index=group.index, dtype='float64')
            
            # Make a copy to avoid modifying the original data
            group = group.copy()
            
            # Ensure group is sorted by 'start_datetime' for rolling to make sense chronologically
            group = group.sort_values('start_datetime')
            
            # Filter out rows with NaT in start_datetime (keep track of original indices)
            original_indices = group.index
            valid_mask = ~pd.isna(group['start_datetime'])
            valid_group = group[valid_mask]
            
            # If the valid group is empty after filtering NaT values, return zeros
            if valid_group.empty:
                return pd.Series(0, index=original_indices, dtype='float64')
            
            try:
                # Set 'start_datetime' as index for rolling operation
                group_indexed = valid_group.set_index('start_datetime')
                
                # Create a series of charge events (1 if charge, 0 otherwise) from the indexed group
                charges_series = (group_indexed['type'] == 'charge').astype(int)
                
                # Calculate rolling sum over 30 days on this series
                rolling_charge_count = charges_series.rolling(window='30D').sum()
                
                # Create a mapping from the datetime index of rolling_charge_count to its values
                datetime_to_count_map = pd.Series(rolling_charge_count.values, index=rolling_charge_count.index)
                
                # Initialize results with zeros for all original indices
                final_counts = pd.Series(0, index=original_indices, dtype='float64')
                
                # For valid entries, map the rolling counts back using start_datetime
                for idx in valid_group.index:
                    dt = valid_group.loc[idx, 'start_datetime']
                    if dt in datetime_to_count_map.index:
                        final_counts.loc[idx] = datetime_to_count_map[dt]
                
                return final_counts
            except Exception as e:
                # If any errors occur, log and return zeros
                print(f"Error in count_charges_last_30d: {str(e)}")
                return pd.Series(0, index=original_indices, dtype='float64')

        data['charge_count_30d'] = data.groupby('user_car_id', group_keys=False).apply(count_charges_last_30d)
        data['charge_count_30d'].fillna(0, inplace=True)

        return data
    
    def _extract_user_features(self, data):
        """
        Extract user-specific features that may be predictive.
        
        Args:
            data: Processed DataFrame
            
        Returns:
            DataFrame with user features
        """
        # Calculate user's typical charging behavior
        user_features = data.groupby('user_car_id').agg({
            'should_charge': 'mean',  # Charging frequency
            'trip_distance_km': ['mean', 'std', 'max'],  # Trip distance patterns
            'trip_duration_hours': ['mean', 'std'],  # Trip duration patterns
            'end_battery_level': ['mean', 'min']  # Battery level patterns
        })
        
        # Flatten the column hierarchy
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        
        # Rename to avoid conflicts
        user_features = user_features.rename(columns={
            'should_charge_mean': 'user_charge_freq',
            'trip_distance_km_mean': 'user_avg_distance',
            'trip_distance_km_std': 'user_std_distance',
            'trip_distance_km_max': 'user_max_distance',
            'trip_duration_hours_mean': 'user_avg_duration',
            'trip_duration_hours_std': 'user_std_duration',
            'end_battery_level_mean': 'user_avg_battery',
            'end_battery_level_min': 'user_min_battery'
        })
        
        # Merge user features back to the original data
        data = data.merge(user_features, left_on='user_car_id', right_index=True, how='left')
        
        return data
    
    def fit_transform(self, data):
        """
        Fit the preprocessor on the data and transform it.
        
        Args:
            data: DataFrame with extracted features
            
        Returns:
            Transformed features as numpy array
        """
        # Fit the preprocessor
        transformed_data = self.preprocessor.fit_transform(data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
        self.fitted = True
        
        # Save the preprocessor if save_dir is specified
        if self.save_dir:
            joblib.dump(self.preprocessor, self.preprocessor_path)
            print(f"Saved preprocessor to {self.preprocessor_path}")
        
        return transformed_data
    
    def transform(self, data):
        """
        Transform data using already fitted preprocessor.
        
        Args:
            data: DataFrame with extracted features
            
        Returns:
            Transformed features as numpy array
        """
        if not self.fitted:
            raise ValueError("Preprocessor is not fitted yet. Call fit_transform first.")
        
        return self.preprocessor.transform(data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])


class EVTripDataset(Dataset):
    """
    Dataset for EV charging prediction with transformer model.
    Creates sequences of trips for each user for input to the transformer.
    """
    
    def __init__(self, data, processor, sequence_length=MAX_SEQUENCE_LENGTH, training=True):
        """
        Initialize the dataset.
        
        Args:
            data: Raw DataFrame containing the EV trip data
            processor: EVDataProcessor instance
            sequence_length: Length of each sequence
            training: Whether this dataset is for training
        """
        self.sequence_length = sequence_length
        self.processor = processor
        self.training = training
        
        # Process the data
        self.processed_data = self.processor.process_data(data)
        
        # Sort by user and datetime
        self.processed_data = self.processed_data.sort_values(['user_car_id', 'start_datetime'])
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        """
        Create sequences of trips for each user.
        
        Returns:
            List of (sequence, target) tuples
        """
        sequences = []
        
        # Group by user
        for user_id, user_data in self.processed_data.groupby('user_car_id'):
            # Skip users with too few records for a sequence
            if len(user_data) <= self.sequence_length:
                print(f"Warning: User {user_id} has only {len(user_data)} records, which is not enough for a sequence of length {self.sequence_length}. Skipping.")
                continue
                
            # Sort by time
            user_data = user_data.sort_values('start_datetime')
            
            # Apply preprocessing
            try:
                if self.training:
                    features = self.processor.fit_transform(user_data)
                else:
                    features = self.processor.transform(user_data)
                
                # Get targets
                targets = user_data[TARGET].values
                
                # Create sequences
                for i in range(len(user_data) - self.sequence_length):
                    seq_features = features[i:i + self.sequence_length]
                    seq_target = targets[i + self.sequence_length]
                    sequences.append((seq_features, seq_target))
            except Exception as e:
                print(f"Error processing data for user {user_id}: {str(e)}")
                continue
        
        # If no sequences could be created, print a warning
        if not sequences:
            print(f"Warning: No sequences could be created! Check your data and sequence length ({self.sequence_length}).")
        else:
            print(f"Created {len(sequences)} sequences for training/prediction.")
            
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        features, target = self.sequences[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32) 