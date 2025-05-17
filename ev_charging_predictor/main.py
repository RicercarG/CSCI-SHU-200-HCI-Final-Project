"""
Main script for EV charging prediction.
Runs the training and prediction pipeline.
"""

import os
import argparse
import pandas as pd
import torch
import numpy as np

from ev_charging_predictor.data.data_loader import EVDataLoader
from ev_charging_predictor.models.transformer_model import TransformerChargingPredictor
from ev_charging_predictor.training.trainer import ModelTrainer
from ev_charging_predictor.utils.metrics import evaluate_and_visualize
from ev_charging_predictor.config import DATA_FILE, MODEL_PATH, MODEL_SAVE_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EV Charging Prediction')
    parser.add_argument('--data', type=str, default=DATA_FILE,
                        help='Path to the CSV file with the data')
    parser.add_argument('--model_dir', type=str, default=MODEL_SAVE_DIR,
                        help='Directory to save model checkpoints')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'online_learning', 'evaluate'],
                        help='Mode to run the script in')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='Path to a saved model to load')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--new_data', type=str, default=None,
                        help='Path to new data for online learning')
    
    return parser.parse_args()


def train(args):
    """
    Train a new model.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'-'*80}\nTraining new model\n{'-'*80}")
    
    # Initialize data loader
    data_loader = EVDataLoader(
        data_path=args.data, 
        processor_save_dir=os.path.join(args.model_dir, 'preprocessor')
    )
    
    # Get train, validation, and test data loaders
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=args.batch_size
    )
    
    # Initialize model and trainer
    model = TransformerChargingPredictor()
    trainer = ModelTrainer(model=model, model_save_dir=args.model_dir)
    
    # Train the model
    trained_model, history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs
    )
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    metrics = trainer.evaluate(test_loader)
    
    # Plot training history
    trained_model.plot_training_history(
        save_path=os.path.join(args.model_dir, 'training_history.png')
    )
    
    return trained_model, metrics


def evaluate(args):
    """
    Evaluate an existing model.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'-'*80}\nEvaluating existing model\n{'-'*80}")
    
    # Initialize data loader
    data_loader = EVDataLoader(
        data_path=args.data, 
        processor_save_dir=os.path.join(args.model_dir, 'preprocessor')
    )
    
    # Get train, validation, and test data loaders
    _, _, test_loader = data_loader.get_dataloaders(
        batch_size=args.batch_size
    )
    
    # Initialize model and load weights
    model = TransformerChargingPredictor()
    model.load(args.model_path)
    
    # Initialize trainer
    trainer = ModelTrainer(model=model, model_save_dir=args.model_dir)
    
    # Evaluate the model
    metrics = trainer.evaluate(test_loader)
    
    return model, metrics


def predict(args):
    """
    Make predictions with an existing model.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'-'*80}\nMaking predictions with existing model\n{'-'*80}")
    
    # Check if we have a model path
    if not args.model_path or not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return None
    
    # Initialize data loader
    data_loader = EVDataLoader(
        data_path=args.data, 
        processor_save_dir=os.path.join(args.model_dir, 'preprocessor')
    )
    
    # Get test data loader
    _, _, test_loader = data_loader.get_dataloaders(
        batch_size=args.batch_size
    )
    
    # Initialize model and load weights
    model = TransformerChargingPredictor()
    model.load(args.model_path)
    
    # Initialize trainer
    trainer = ModelTrainer(model=model, model_save_dir=args.model_dir)
    
    # Make predictions
    predictions, probabilities = trainer.predict(test_loader)
    
    # Extract users and their latest trip data from the processed data
    unique_users = test_loader.dataset.processed_data['user_car_id'].unique()
    latest_trips = []
    
    for user_id in unique_users:
        user_data = test_loader.dataset.processed_data[test_loader.dataset.processed_data['user_car_id'] == user_id]
        if len(user_data) > 0:
            # Get the most recent trip
            latest_trip = user_data.sort_values('start_datetime').iloc[-1]
            latest_trips.append(latest_trip)
    
    # Create results dataframe
    if len(latest_trips) == len(predictions):
        results = pd.DataFrame(latest_trips)
        results['predicted_should_charge'] = predictions
        results['charging_probability'] = probabilities
        
        # Select columns for output
        output_columns = ['user_car_id', 'start_datetime', 'end_battery_level', 
                         'type', 'predicted_should_charge', 'charging_probability']
        results = results[output_columns]
        
        # Add a recommendation column
        results['recommendation'] = results['predicted_should_charge'].apply(
            lambda x: "CHARGE" if x == 1 else "DO NOT CHARGE"
        )
        
        # Format probabilities as percentages
        results['confidence'] = (results['charging_probability'] * 100).round(1).astype(str) + '%'
        
        # Save results
        results_path = os.path.join(args.model_dir, 'predictions.csv')
        results.to_csv(results_path, index=False)
        print(f"Predictions saved to {results_path}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(f"Total users: {len(results)}")
        print(f"Users recommended to charge: {results['predicted_should_charge'].sum()}")
        print(f"Average charging probability: {results['charging_probability'].mean():.2f}")
        
        # Display compact version of results
        compact_results = results[['user_car_id', 'end_battery_level', 'recommendation', 'confidence']]
        print("\nUser Recommendations:")
        print(compact_results.head(10).to_string(index=False))
        if len(compact_results) > 10:
            print(f"... and {len(compact_results) - 10} more users")
    else:
        print(f"Error: Mismatch between number of latest trips ({len(latest_trips)}) and predictions ({len(predictions)})")
        results = pd.DataFrame({
            'predictions': predictions,
            'probabilities': probabilities
        })
        results_path = os.path.join(args.model_dir, 'predictions_raw.csv')
        results.to_csv(results_path, index=False)
        print(f"Raw predictions saved to {results_path}")
    
    return results


def online_learning(args):
    """
    Update an existing model with new data.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'-'*80}\nUpdating model with new data\n{'-'*80}")
    
    # Check if we have a model path and new data
    if not args.model_path or not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return None
    
    if not args.new_data or not os.path.exists(args.new_data):
        print(f"Error: New data path {args.new_data} does not exist.")
        return None
    
    # Initialize data loader for the original data
    data_loader = EVDataLoader(
        data_path=args.data, 
        processor_save_dir=os.path.join(args.model_dir, 'preprocessor')
    )
    
    # Load the model
    model = TransformerChargingPredictor()
    model.load(args.model_path)
    
    # Initialize trainer
    trainer = ModelTrainer(model=model, model_save_dir=args.model_dir)
    
    # Load new data
    new_data = pd.read_csv(args.new_data)
    print(f"Loaded {len(new_data)} new records")
    
    # Create a data loader for the new data
    new_data_loader = data_loader.get_online_update_loader(
        new_data=new_data, 
        batch_size=args.batch_size
    )
    
    # Update the model
    updated_model, history = trainer.online_learning(
        new_data_loader=new_data_loader,
        epochs=args.epochs
    )
    
    # Get test data loader for evaluation
    _, _, test_loader = data_loader.get_dataloaders(
        batch_size=args.batch_size
    )
    
    # Evaluate the updated model
    print("\nEvaluating updated model on test data...")
    metrics = trainer.evaluate(test_loader)
    
    return updated_model, metrics


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Run the appropriate mode
    if args.mode == 'train':
        model, metrics = train(args)
    elif args.mode == 'evaluate':
        model, metrics = evaluate(args)
    elif args.mode == 'predict':
        results = predict(args)
    elif args.mode == 'online_learning':
        model, metrics = online_learning(args)
    else:
        print(f"Error: Invalid mode {args.mode}")
        return
    
    print("\nDone!")


if __name__ == "__main__":
    main() 