"""
Transformer-based model for EV charging prediction.
Uses a state-of-the-art Transformer architecture for time series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .base_model import BaseModel
from ..config import (
    D_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS, 
    DROPOUT, DIM_FEEDFORWARD, LEARNING_RATE,
    ONLINE_LEARNING_RATE, EARLY_STOPPING_PATIENCE
)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer model.
    Adds information about the position of tokens in the sequence.
    """
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
        self.register_buffer('positional_encoding', self.encoding)
    
    def forward(self, x):
        """Add positional encoding to the input."""
        return x + self.positional_encoding[:, :x.size(1)].detach()


class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction.
    Uses self-attention to capture temporal patterns in EV usage data.
    """
    
    def __init__(self, input_dim, d_model=D_MODEL, nhead=NUM_HEADS, 
                 num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, 
                 dropout=DROPOUT):
        super(TransformerModel, self).__init__()
        
        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # Attention pooling
        self.query = nn.Parameter(torch.randn(d_model))
        self.attention_fc = nn.Linear(d_model, d_model)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)
        
    def forward(self, x, src_mask=None):
        """
        Forward pass through the transformer model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            src_mask: Optional mask for the encoder
            
        Returns:
            Output tensor of shape [batch_size]
        """
        # Replace any NaN values with zeros to avoid numerical issues
        x = torch.nan_to_num(x, nan=0.0)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_mask)
        
        # Attention pooling
        attention_scores = torch.matmul(
            F.tanh(self.attention_fc(x)), 
            self.query
        )
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        # Final classification layers
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Ensure output is between 0 and 1
        # Use torch.clamp to ensure we don't have any out-of-range values
        return torch.clamp(torch.sigmoid(x.squeeze(-1)), min=0.0, max=1.0)


class TransformerChargingPredictor(BaseModel):
    """
    Transformer-based model for EV charging prediction.
    Implements the base model interface.
    """
    
    def __init__(self, input_dim=None, device=None):
        """
        Initialize the model.
        
        Args:
            input_dim: Dimension of input features
            device: Device to run the model on (cpu or cuda)
        """
        self.input_dim = input_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        
        print(f"Using device: {self.device}")
    
    def _initialize_model(self, input_dim):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Dimension of input features
        """
        self.input_dim = input_dim
        self.model = TransformerModel(input_dim=input_dim)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    def fit(self, train_loader, val_loader=None, epochs=20, callbacks=None):
        """
        Train the model on the provided data.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train for
            callbacks: List of callback functions
            
        Returns:
            History of training metrics
        """
        # Initialize model if not already initialized
        if self.model is None:
            # Get input dimension from the first batch
            for batch_features, _ in train_loader:
                input_dim = batch_features.shape[2]
                self._initialize_model(input_dim)
                break
        
        # Create history dictionary
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_auc": []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch_features, batch_labels in progress_bar:
                # Handle NaN values in features and labels
                batch_features = torch.nan_to_num(batch_features, nan=0.0)
                batch_labels = torch.nan_to_num(batch_labels, nan=0.0).clamp(0.0, 1.0)
                
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                
                # Ensure outputs are valid for BCE loss
                outputs = torch.clamp(outputs, min=1e-7, max=1.0 - 1e-7)
                
                # Calculate loss with epsilon to prevent log(0)
                try:
                    loss = F.binary_cross_entropy(outputs, batch_labels)
                except Exception as e:
                    print(f"Error in loss calculation: {e}")
                    print(f"Outputs min/max: {outputs.min().item()}, {outputs.max().item()}")
                    print(f"Labels min/max: {batch_labels.min().item()}, {batch_labels.max().item()}")
                    # Use a fallback loss if necessary
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss value: {loss.item()}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update progress bar
                train_losses.append(loss.item())
                progress_bar.set_postfix({"loss": np.mean(train_losses)})
            
            # Calculate average training loss
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
                # Update history
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])
                history["val_auc"].append(val_metrics["auc"])
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics["loss"])
                
                # Early stopping check
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    # Save the best model
                    self.save("best_model.pt")
                else:
                    patience_counter += 1
                
                # Print epoch summary
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")
                
                # Check for early stopping
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Load the best model if validation was used
        if val_loader is not None:
            self.load("best_model.pt")
        
        self.history = history
        return history
    
    def partial_fit(self, data_loader, epochs=1):
        """
        Update the model with new data (online learning).
        
        Args:
            data_loader: DataLoader containing the new data
            epochs: Number of epochs to train for
            
        Returns:
            History of training metrics
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call fit first.")
        
        # Set a lower learning rate for online updates
        for param_group in self.optimizer.param_groups:
            original_lr = param_group['lr']
            param_group['lr'] = ONLINE_LEARNING_RATE
        
        # Training loop for online learning
        history = {"train_loss": []}
        
        self.model.train()
        for epoch in range(epochs):
            train_losses = []
            
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                
                # Calculate loss
                loss = F.binary_cross_entropy(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track loss
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            print(f"Online Update Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")
        
        # Restore original learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = original_lr
        
        return history
    
    def predict(self, data_loader):
        """
        Make predictions using the trained model.
        
        Args:
            data_loader: DataLoader containing the data to predict on
            
        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call fit first.")
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch_features, _ in data_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                predictions = (outputs > 0.5).int().cpu().numpy()
                all_predictions.extend(predictions)
        
        return np.array(all_predictions)
    
    def predict_proba(self, data_loader):
        """
        Predict probability of charging.
        
        Args:
            data_loader: DataLoader containing the data to predict on
            
        Returns:
            Array of probabilities
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call fit first.")
        
        self.model.eval()
        all_probas = []
        
        with torch.no_grad():
            for batch_features, _ in data_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                all_probas.extend(outputs.cpu().numpy())
        
        return np.array(all_probas)
    
    def evaluate(self, data_loader, metrics=None):
        """
        Evaluate the model performance.
        
        Args:
            data_loader: DataLoader containing the data to evaluate on
            metrics: List of metric functions
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call fit first.")
        
        self.model.eval()
        all_outputs = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                loss = F.binary_cross_entropy(outputs, batch_labels)
                
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_losses.append(loss.item())
        
        # Convert to numpy arrays
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        
        # Calculate binary predictions
        all_predictions = (all_outputs > 0.5).astype(int)
        
        # Calculate metrics
        results = {
            "loss": np.mean(all_losses),
            "accuracy": accuracy_score(all_labels, all_predictions),
            "precision": precision_score(all_labels, all_predictions, zero_division=0),
            "recall": recall_score(all_labels, all_predictions, zero_division=0),
            "f1": f1_score(all_labels, all_predictions, zero_division=0),
            "auc": roc_auc_score(all_labels, all_outputs) if len(np.unique(all_labels)) > 1 else 0.5
        }
        
        return results
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Nothing to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save model state
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'input_dim': self.input_dim,
            'history': self.history
        }
        
        torch.save(state, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        # Load model state
        state = torch.load(path, map_location=self.device)
        
        # Initialize model if needed
        if self.model is None:
            self._initialize_model(state['input_dim'])
        
        # Load state dictionaries
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if state['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        self.input_dim = state['input_dim']
        self.history = state.get('history', self.history)
        
        print(f"Model loaded from {path}")
    
    def plot_training_history(self, save_path=None):
        """
        Plot the training history.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.history["train_loss"]:
            print("No training history available.")
            return
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot loss
        axs[0].plot(self.history["train_loss"], label="Train Loss")
        if self.history.get("val_loss"):
            axs[0].plot(self.history["val_loss"], label="Validation Loss")
        axs[0].set_title("Loss During Training")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot metrics
        if self.history.get("val_accuracy"):
            axs[1].plot(self.history["val_accuracy"], label="Accuracy")
        if self.history.get("val_precision"):
            axs[1].plot(self.history["val_precision"], label="Precision")
        if self.history.get("val_recall"):
            axs[1].plot(self.history["val_recall"], label="Recall")
        if self.history.get("val_f1"):
            axs[1].plot(self.history["val_f1"], label="F1 Score")
        if self.history.get("val_auc"):
            axs[1].plot(self.history["val_auc"], label="AUC")
        
        axs[1].set_title("Validation Metrics During Training")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Score")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        # plt.show() 