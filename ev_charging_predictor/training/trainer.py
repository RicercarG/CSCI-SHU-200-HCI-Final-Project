"""
Trainer module for EV charging prediction.
Handles model training, evaluation, and saving/loading.
"""

import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

from ..models.transformer_model import TransformerChargingPredictor
from ..utils.metrics import evaluate_and_visualize
from ..config import (
    MODEL_PATH, NUM_EPOCHS, EARLY_STOPPING_PATIENCE
)


class ModelTrainer:
    """
    Handles the training and evaluation of EV charging prediction models.
    """
    
    def __init__(self, model=None, model_save_dir=None):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train (if None, a new TransformerChargingPredictor will be used)
            model_save_dir: Directory to save model checkpoints
        """
        self.model = model or TransformerChargingPredictor()
        self.model_save_dir = model_save_dir
        
        if model_save_dir:
            os.makedirs(model_save_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader=None, epochs=NUM_EPOCHS, 
              patience=EARLY_STOPPING_PATIENCE, save_best=True):
        """
        Train the model on the provided data.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train for
            patience: Patience for early stopping
            save_best: Whether to save the best model
            
        Returns:
            Trained model and history
        """
        print(f"Starting training for {epochs} epochs with early stopping patience {patience}")
        start_time = time.time()
        
        history = self.model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            callbacks=None  # Could add custom callbacks here
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        if save_best:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join(
                self.model_save_dir or ".", 
                f"ev_charging_model_{timestamp}.pt"
            )
            self.model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
        
        return self.model, history
    
    def evaluate(self, test_loader, save_results=True):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating model on test data...")
        
        # Calculate metrics and generate visualizations
        if save_results and self.model_save_dir:
            save_dir = os.path.join(self.model_save_dir, "evaluation")
        else:
            save_dir = None
            
        metrics = evaluate_and_visualize(self.model, test_loader, save_dir=save_dir)
        
        return metrics
    
    def online_learning(self, new_data_loader, epochs=1):
        """
        Update the model with new data (online learning).
        
        Args:
            new_data_loader: DataLoader containing the new data
            epochs: Number of epochs to train for
            
        Returns:
            Updated model and history
        """
        print(f"Performing online learning update with {len(new_data_loader.dataset)} samples")
        
        history = self.model.partial_fit(new_data_loader, epochs=epochs)
        
        # Save updated model
        if self.model_save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join(
                self.model_save_dir, 
                f"ev_charging_model_updated_{timestamp}.pt"
            )
            self.model.save(model_save_path)
            print(f"Updated model saved to {model_save_path}")
        
        return self.model, history
    
    def predict(self, data_loader):
        """
        Make predictions using the trained model.
        
        Args:
            data_loader: DataLoader containing the data to predict on
            
        Returns:
            Array of predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please load or train a model first.")
        
        print("Making predictions on test data...")
        
        # Get predictions
        all_predictions = []
        all_probabilities = []
        
        # Set model to evaluation mode
        self.model.model.eval()  # Access the PyTorch model inside TransformerChargingPredictor
        
        with torch.no_grad():
            for batch_features, _ in data_loader:
                batch_features = batch_features.to(self.model.device)
                
                # Handle possible NaN values
                batch_features = torch.nan_to_num(batch_features, nan=0.0)
                
                # Get model outputs
                outputs = self.model.model(batch_features)
                
                # Convert outputs to predictions
                batch_predictions = (outputs > 0.5).int().cpu().numpy()
                batch_probabilities = outputs.cpu().numpy()
                
                # Add to our collection
                all_predictions.extend(batch_predictions)
                all_probabilities.extend(batch_probabilities)
        
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        
        print(f"Made {len(predictions)} predictions.")
        
        return predictions, probabilities
    
    def save_model(self, path=None):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model (if None, use default path)
        """
        save_path = path or os.path.join(self.model_save_dir or ".", "ev_charging_model.pt")
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, path=None):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from (if None, use default path)
        """
        load_path = path or os.path.join(self.model_save_dir or ".", "ev_charging_model.pt")
        self.model.load(load_path)
        print(f"Model loaded from {load_path}")
        return self.model 