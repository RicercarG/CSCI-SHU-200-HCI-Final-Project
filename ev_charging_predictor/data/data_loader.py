"""
Data loader module for EV charging prediction.
Handles loading data from files and splitting into train/validation/test sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import os

from .data_processor import EVDataProcessor, EVTripDataset
from ..config import (
    DATA_FILE, VALIDATION_SPLIT, TEST_SPLIT, 
    BATCH_SIZE, MAX_SEQUENCE_LENGTH
)

class EVDataLoader:
    """
    Handles loading and processing EV trip data for model training and evaluation.
    """
    
    def __init__(self, data_path=DATA_FILE, processor_save_dir=None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing the data
            processor_save_dir: Directory to save/load preprocessors
        """
        self.data_path = data_path
        self.processor = EVDataProcessor(save_dir=processor_save_dir)
        
        # Load the data
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        print(f"Loaded {len(self.data)} records.")
    
    def prepare_datasets(self, val_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, 
                         sequence_length=MAX_SEQUENCE_LENGTH, random_state=42):
        """
        Prepare train, validation, and test datasets.
        
        Args:
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            sequence_length: Length of sequences for the transformer model
            random_state: Random seed for reproducibility
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # Split users rather than individual trips to avoid data leakage
        unique_users = self.data['user_car_id'].unique()
        
        # First split: separate test set
        users_train_val, users_test = train_test_split(
            unique_users, 
            test_size=test_split, 
            random_state=random_state
        )
        
        # Second split: separate validation set from training set
        users_train, users_val = train_test_split(
            users_train_val,
            test_size=val_split / (1 - test_split),
            random_state=random_state
        )
        
        # Filter data by user
        train_data = self.data[self.data['user_car_id'].isin(users_train)]
        val_data = self.data[self.data['user_car_id'].isin(users_val)]
        test_data = self.data[self.data['user_car_id'].isin(users_test)]
        
        print(f"Train data: {len(train_data)} records from {len(users_train)} users")
        print(f"Validation data: {len(val_data)} records from {len(users_val)} users")
        print(f"Test data: {len(test_data)} records from {len(users_test)} users")
        
        # Create datasets
        train_dataset = EVTripDataset(train_data, self.processor, sequence_length, training=True)
        val_dataset = EVTripDataset(val_data, self.processor, sequence_length, training=False)
        test_dataset = EVTripDataset(test_data, self.processor, sequence_length, training=False)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(self, batch_size=BATCH_SIZE):
        """
        Create DataLoader objects for train, validation, and test datasets.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_online_update_loader(self, new_data, batch_size=BATCH_SIZE):
        """
        Create a DataLoader for online learning with new data.
        
        Args:
            new_data: DataFrame containing new data
            batch_size: Batch size for training
            
        Returns:
            DataLoader for the new data
        """
        online_dataset = EVTripDataset(new_data, self.processor, training=False)
        
        online_loader = DataLoader(
            online_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        return online_loader 