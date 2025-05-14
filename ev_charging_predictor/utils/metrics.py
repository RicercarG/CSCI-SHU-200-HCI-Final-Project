"""
Metrics utilities for evaluating EV charging prediction models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
import seaborn as sns
import pandas as pd


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate common classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_proba: Predicted probabilities (for AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add AUC if probabilities are provided
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
    
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print a formatted classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of target class names
    """
    if target_names is None:
        target_names = ["No Charge", "Charge"]
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:")
    print(report)


def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Plot the ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true, y_proba, save_path=None):
    """
    Plot the Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['No Charge', 'Charge'])
    plt.yticks([0.5, 1.5], ['No Charge', 'Charge'])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=20, save_path=None):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance scores
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    # Create a DataFrame for easier sorting and plotting
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and take top N
    top_features = feature_df.sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def evaluate_and_visualize(model, test_loader, feature_names=None, save_dir=None):
    """
    Comprehensive evaluation of a model with visualizations.
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        feature_names: List of feature names (for feature importance if available)
        save_dir: Directory to save plots
    """
    # Get predictions
    y_pred = model.predict(test_loader)
    y_proba = model.predict_proba(test_loader)
    
    # Get true labels
    y_true = []
    for _, batch_labels in test_loader:
        y_true.extend(batch_labels.numpy())
    y_true = np.array(y_true)
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")
    
    print_classification_report(y_true, y_pred)
    
    # Plot visualizations
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, 
        save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None
    )
    
    # ROC curve
    plot_roc_curve(
        y_true, y_proba, 
        save_path=f"{save_dir}/roc_curve.png" if save_dir else None
    )
    
    # Precision-Recall curve
    plot_precision_recall_curve(
        y_true, y_proba, 
        save_path=f"{save_dir}/pr_curve.png" if save_dir else None
    )
    
    # Feature importance if available (e.g., for tree-based models)
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        plot_feature_importance(
            feature_names, model.feature_importances_,
            save_path=f"{save_dir}/feature_importance.png" if save_dir else None
        )
    
    return metrics 