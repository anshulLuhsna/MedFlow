"""
ML Core Configuration
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR.parent / "data"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "demand_forecaster").mkdir(exist_ok=True)
(MODELS_DIR / "shortage_detector").mkdir(exist_ok=True)
(MODELS_DIR / "preference_learner").mkdir(exist_ok=True)

# Resource types
RESOURCE_TYPES = ["ventilators", "o2_cylinders", "beds", "medications", "ppe"]

# Demand Forecasting Configuration
DEMAND_FORECAST_CONFIG = {
    "sequence_length": 30,      # Use 30 days of history
    "forecast_horizon": 14,     # Predict 14 days ahead
    "lstm_units": 128,
    "lstm_layers": 2,
    "dropout": 0.5,             # INCREASED from 0.2 → 0.3 → 0.4 → 0.5 for MC Dropout calibration
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 15,  # Increased from 10 to allow more training
    "early_stopping_min_delta": 0.001,  # Require at least 0.1% improvement
    "learning_rate": 0.001,
    "validation_split": 0.2
}

# Shortage Detection Configuration
SHORTAGE_DETECTION_CONFIG = {
    "risk_levels": ["low", "medium", "high", "critical"],
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "class_weight": "balanced"  # Handle imbalanced classes
}

# Optimization Configuration
OPTIMIZATION_CONFIG = {
    "solver": "PULP_CBC_CMD",  # Can use GLPK, COIN, etc.
    "time_limit": 30,          # Max 30 seconds per optimization
    "objectives": {
        "minimize_shortage": 1.0,
        "minimize_cost": 0.3,
        "maximize_coverage": 0.8,
        "fairness": 0.5
    },
    "transfer_cost_per_unit": {
        "ventilators": 500,
        "o2_cylinders": 50,
        "beds": 0,  # Beds can't be transferred
        "medications": 10,
        "ppe": 5
    },
    "max_transfer_distance_km": 200
}

# Preference Learning Configuration
PREFERENCE_LEARNING_CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "learning_rate": 0.1,
    "feature_weights": {
        "cost_score": 0.33,
        "speed_score": 0.33,
        "coverage_score": 0.34
    },
    "llm_model": "claude-sonnet-4-20250514",
    "embedding_model": "sentence-transformers/all-mpnet-base-v2"
}

# Feature Engineering
FEATURE_CONFIG = {
    "days_of_history": 30,
    "aggregation_windows": [7, 14, 30],  # Weekly, bi-weekly, monthly
    "seasonal_features": True,
    "trend_features": True,
    "lag_features": [1, 7, 14, 30]
}

# Model Paths
MODEL_PATHS = {
    "demand_forecaster": MODELS_DIR / "demand_forecaster",
    "shortage_detector": MODELS_DIR / "shortage_detector",
    "preference_learner": MODELS_DIR / "preference_learner"
}

# Training Configuration
TRAINING_CONFIG = {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42
}