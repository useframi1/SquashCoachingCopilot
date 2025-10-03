# Rally State Prediction Pipeline

A clean, modular pipeline for predicting squash rally states (start, active, end) from player tracking data. Supports both rule-based and ML-based models through a unified interface.

## Overview

This pipeline provides two approaches for rally state prediction:

1. **Rule-Based Model**: Uses position constraints and distance thresholds with temporal requirements
2. **ML-Based Model**: Uses machine learning with engineered features from annotated data

Both models implement the same interface, making them interchangeable through configuration.

## Model Types

### Rule-Based Model

Clean rule-based logic following squash game rules:

- **Start State**: Starting position constraints met for 50+ frames (both players behind service line, opposite sides, at least one in service box) + distance between 2-3m
- **Active State**: Players in close range (distance ≤ 2m)  
- **End State**: Distance above 3m threshold for 50+ consecutive frames
- **Transition Rules**: start→active, active→end, end→start only

### ML-Based Model

Machine learning approach using feature engineering:

1. **Annotation**: Annotate videos with rally states, aggregate base metrics every 50 frames
2. **Feature Engineering**: Convert base metrics to 21 ML features
3. **Training**: Train on videos with group-aware split
4. **Inference**: Real-time prediction using the same feature engineering

## Pipeline Components

### 1. Configuration (`config.py`)

Centralized configuration for all components:
- Model selection (rule_based vs ml_based)
- Rule-based model parameters (distance thresholds, temporal windows)
- ML model paths and parameters

### 2. Base Model Interface (`modeling/base_model.py`)

Abstract base class ensuring both models implement:
- `predict(df)`: Batch prediction method
- `reset_state()`: Reset internal state

### 3. Rule-Based Model (`modeling/rule-based/rule_based_model.py`)

Implements game logic with configurable parameters:
- Position constraint checking
- Distance-based state classification
- Temporal window requirements
- Valid state transition enforcement

### 4. ML-Based Model (`modeling/ml-based/ml_based_model.py`)

Machine learning pipeline:
- Feature engineering integration
- XGBoost/Random Forest training
- Model loading for inference
- Batch prediction support

### 5. State Predictor (`modeling/predictor.py`)

Model-agnostic predictor that:
- Automatically loads the active model from config
- Provides unified interface for both model types
- Supports single and batch predictions

## Features

The pipeline generates 21 features from 5 base metrics:

**Base Metrics** (from your annotations):

-   `mean_distance`
-   `median_player1_x`
-   `median_player1_y`
-   `median_player2_x`
-   `median_player2_y`

**Engineered Features**:

1. `mean_distance`
2. `distance_lag_1`, `distance_lag_2`, `distance_lag_3`
3. `distance_change`, `distance_acceleration`
4. `distance_rolling_mean`, `distance_rolling_std`, `distance_rolling_min`, `distance_rolling_max`
5. `median_player1_x`, `median_player1_y`, `median_player2_x`, `median_player2_y`
6. `player1_movement`, `player2_movement`
7. `player1_court_side`, `player2_court_side`, `players_same_side`
8. `player1_from_service_line`, `player2_from_service_line`

## Usage

### Model Selection

Switch between models in `config.py`:

```python
CONFIG = {
    "active_model": "rule_based",  # or "ml_based"
    # ... other settings
}
```

### Training an ML Model

```python
from modeling.ml_based.ml_based_model import MLBasedModel

trainer = MLBasedModel()
accuracy = trainer.run_training_pipeline()
```

### Using the StatePredictor

The StatePredictor automatically uses the active model from config:

```python
from modeling.predictor import StatePredictor

predictor = StatePredictor()  # Loads rule_based or ml_based based on config

# Single prediction
base_metrics = {
    'mean_distance': 2.1,
    'median_player1_x': 2.3,
    'median_player1_y': 6.1,
    'median_player2_x': 4.1,
    'median_player2_y': 5.9
}

prediction = predictor.predict_single(base_metrics)
print(f"Predicted state: {prediction}")

# Batch prediction
import pandas as pd
df = pd.read_csv("new_video_metrics.csv")
df_pred = predictor.predict_batch(df)
print(df_pred[['frame_number', 'predicted_state']])
```

### Direct Model Usage

You can also use models directly:

```python
# Rule-based model
from modeling.rule_based.rule_based_model import RuleBasedModel
rule_model = RuleBasedModel()
result = rule_model.predict(df)

# ML-based model  
from modeling.ml_based.ml_based_model import MLBasedModel
ml_model = MLBasedModel()
result = ml_model.predict(df)
```

### Real-time Inference Simulation

```python
from predictor import StatePredictor, InferenceMetricsAggregator

predictor = StatePredictor()
aggregator = InferenceMetricsAggregator()

# For each video frame:
for frame_idx in range(video_length):
    # Get player positions from tracking
    player1_pos, player2_pos = get_player_positions(frame)

    # Add to aggregator
    aggregator.add_frame_metrics(player1_pos, player2_pos)

    # Predict every 50 frames
    if aggregator.should_predict():
        base_metrics = aggregator.get_window_metrics()
        prediction = predictor.predict_single(base_metrics)
        print(f"Frame {frame_idx}: {prediction}")
```

## Configuration

Edit `config.py` to customize:

```python
CONFIG = {
    "active_model": "rule_based",  # or "ml_based"
    "window_size": 50,
    "modeling": {
        "ml_based": {
            "model_path": "models/rally_state_model.pkl",
            "model_type": "xgboost",  # or "random_forest"
        },
        "rule_based": {
            "state_window_frames": 50,  # Frames required for state transitions
            "start_state": {
                "distance_min": 2.0,  # meters
                "distance_max": 3.0,  # meters
            },
            "active_state": {
                "distance_max": 2.0,  # meters
            },
            "end_state": {
                "distance_min": 3.0,  # meters
                "consecutive_frames": 50,  # frames above threshold required
            },
            # ... court configuration
        },
        "test_size": 0.2,
    },
    # ... other settings
}
```

## File Structure

```
├── config.py                              # Configuration
├── modeling/
│   ├── base_model.py                      # Abstract base model interface
│   ├── predictor.py                       # Model-agnostic predictor
│   ├── rule_based/
│   │   └── rule_based_model.py           # Rule-based implementation
│   └── ml_based/
│       └── ml_based_model.py             # ML-based implementation
├── utilities/
│   └── feature_engineer.py               # Feature engineering logic
├── testing/
│   └── evaluator.py                      # Evaluation pipeline
└── data/
    └── annotations/                      # CSV files with base metrics
```

## Key Benefits

-   **Model Agnostic**: Switch between rule-based and ML-based models without code changes
-   **Unified Interface**: Both models implement the same `predict()` method
-   **Clean Rule-Based Logic**: Simplified rules following squash game constraints
-   **Transition Enforcement**: Proper state transition rules (start→active→end→start)
-   **Configurable**: Easily adjust thresholds, temporal windows, and model selection
-   **Modular Design**: Easy to debug, test, and maintain each model separately

## Rule-Based Model Features

-   **Position Constraints**: Enforces squash serving rules (service line, service boxes)
-   **Temporal Requirements**: Requires sustained conditions (50 frames) before state changes
-   **Distance Thresholds**: Configurable distance ranges for each state
-   **Valid Transitions**: Prevents invalid state jumps (e.g., active→start)
-   **Stateful Tracking**: Maintains frame counters for temporal logic
