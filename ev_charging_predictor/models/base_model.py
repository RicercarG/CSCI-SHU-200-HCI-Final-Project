"""
Base model class for EV charging prediction.
Defines the interface that all models should implement.
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for EV charging prediction models.
    Defines the interface that all models should implement.
    """
    
    @abstractmethod
    def fit(self, train_loader, val_loader=None, epochs=1, callbacks=None):
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
        pass
    
    @abstractmethod
    def predict(self, data_loader):
        """
        Make predictions using the trained model.
        
        Args:
            data_loader: DataLoader containing the data to predict on
            
        Returns:
            Array of predictions (0 or 1)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, data_loader):
        """
        Predict probability of charging.
        
        Args:
            data_loader: DataLoader containing the data to predict on
            
        Returns:
            Array of probabilities
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_loader, metrics=None):
        """
        Evaluate the model performance.
        
        Args:
            data_loader: DataLoader containing the data to evaluate on
            metrics: List of metric functions
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    @abstractmethod
    def partial_fit(self, data_loader, epochs=1):
        """
        Update the model with new data (online learning).
        
        Args:
            data_loader: DataLoader containing the new data
            epochs: Number of epochs to train for
            
        Returns:
            History of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass 