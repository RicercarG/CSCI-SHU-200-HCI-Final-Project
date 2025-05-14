# EV Charging Predictor

An advanced machine learning system that predicts when electric vehicle users should charge their cars, based on their historical usage patterns. The system uses state-of-the-art Transformer models and supports online learning to continuously improve predictions as new data becomes available.

## Features

- **State-of-the-art Transformer Model**: Leverages the power of self-attention mechanisms to capture complex temporal patterns in EV usage data
- **Online Learning**: Continuously updates the model with new user data without retraining from scratch
- **Advanced Feature Engineering**: Extracts meaningful features from raw trip data:
  - Time-based patterns (daily, weekly)
  - Trip characteristics (duration, distance)
  - Battery consumption patterns
  - Weather impacts
- **User-Specific Patterns**: Learns individual user behavior patterns for personalized charging recommendations
- **Comprehensive Evaluation**: Detailed metrics and visualizations to understand model performance

## Project Structure

```
ev_charging_predictor/
├── data/
│   ├── data_loader.py       # Handles loading and splitting data
│   └── data_processor.py    # Feature engineering and preprocessing
├── models/
│   ├── base_model.py        # Abstract base class for models
│   └── transformer_model.py # Transformer-based implementation
├── training/
│   └── trainer.py           # Training logic and evaluation
├── utils/
│   └── metrics.py           # Evaluation metrics and visualization
├── config.py                # Configuration settings
├── main.py                  # Entry point script
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ev-charging-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a New Model

```bash
python -m ev_charging_predictor.main --mode train --data path/to/data.csv --model_path models/ --epochs 20
```

### Making Predictions

```bash
python -m ev_charging_predictor.main --mode predict --data path/to/data.csv --model_path models/ev_charging_model.pt
```

### Online Learning with New Data

```bash
python -m ev_charging_predictor.main --mode online_learning --data path/to/original_data.csv --model_path models/ev_charging_model.pt --new_data path/to/new_data.csv
```

### Evaluating a Model

```bash
python -m ev_charging_predictor.main --mode evaluate --data path/to/data.csv --model_path models/ev_charging_model.pt
```

## Transformer Model Architecture

The model uses a sophisticated Transformer architecture that includes:

1. **Input Embedding**: Projects raw features into a high-dimensional embedding space
2. **Positional Encoding**: Adds information about the position of trips in the sequence
3. **Multi-Head Self-Attention**: Captures dependencies between trips regardless of their distance in the sequence
4. **Feed-Forward Networks**: Adds non-linearity and transforms the representations
5. **Attention Pooling**: Weighting the importance of different trips for the final prediction

This architecture excels at capturing complex temporal patterns and user-specific behaviors, making it ideal for predicting charging needs.

## Online Learning Implementation

The system implements online learning through:

1. **Incremental Updates**: The model can be updated with new data without retraining from scratch
2. **Adaptive Learning Rate**: Uses a lower learning rate for online updates to prevent catastrophic forgetting
3. **Contextual Awareness**: Maintains knowledge of historical patterns while adapting to new behaviors

## Input Data Format

The system expects data in CSV format with the following fields:

```
history_id,user_car_id,username,car_model,type,start_date,start_time,start_location_latitude,start_location_longitude,end_date,end_time,end_location_latitude,end_location_longitude,weather,paid,end_battery_level
```

## Performance Metrics

The system evaluates performance using:

- Accuracy
- Precision & Recall
- F1 Score
- ROC AUC
- Confusion Matrix

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 