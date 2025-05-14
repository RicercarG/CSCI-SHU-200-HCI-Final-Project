"""
Configuration settings for the EV charging prediction system.
"""

import os

# Data settings
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached_table")
DATA_FILE = os.path.join(DATA_DIR, "user_car_history_table.csv")
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Model settings
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "transformer_ev_model.pt")

# Training settings
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 5

# Transformer model settings
MAX_SEQUENCE_LENGTH = 10
D_MODEL = 128
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Features
CATEGORICAL_FEATURES = ['weather', 'day_of_week', 'time_of_day']
NUMERICAL_FEATURES = ['end_battery_level', 'trip_duration_hours', 'trip_distance_km']

# Target
TARGET = 'should_charge'

# Online learning settings
UPDATE_FREQUENCY = 100  # Update model every N new samples
ONLINE_LEARNING_RATE = 5e-5  # Lower learning rate for online updates 