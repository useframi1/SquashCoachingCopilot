# Rally State Prediction Pipeline

A clean, modular pipeline for predicting squash rally states (start, active, end) from player tracking data.

## Overview

This pipeline follows your exact workflow:

1. **Annotation**: Annotate 5 videos with rally states, aggregate base metrics every 50 frames
2. **Feature Engineering**: Convert base metrics to 21 ML features
3. **Training**: Train on 4 videos, test on 1 video (group-aware split)
4. **Inference**: Real-time prediction using the same feature engineering

## Pipeline Components

### 1. Configuration (`config.py`)

-   Centralized configuration for all components
-   Easily adjustable model types, paths, and parameters

### 2. Feature Engineering (`feature_engineer.py`)

-   Converts 5 base metrics to 21 engineered features
-   Handles temporal features (lags, rolling stats, movement)
-   Consistent processing for training and inference

### 3. Model Training (`model_trainer.py`)

-   Loads annotated CSV files with base metrics
-   Engineers features grouped by video
-   Trains XGBoost or Random Forest models
-   Video-aware train/test split to prevent data leakage

### 4. Inference (`predictor.py`)

-   `StatePredictor`: Makes predictions using trained model
-   `InferenceMetricsAggregator`: Aggregates base metrics from video frames
-   Consistent feature engineering with training pipeline

### 5. Evaluation (`evaluator.py`)

-   Evaluates trained models on test data
-   Generates accuracy metrics and visualizations
-   Ensures consistency between training and inference

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

### Training a Model

```python
from model_trainer import ModelTrainer

trainer = ModelTrainer()
accuracy = trainer.run_training_pipeline()
```

### Evaluating a Model

```python
from evaluator import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.run_evaluation(video_filter="video-2")
```

### Single Prediction

```python
from predictor import StatePredictor

predictor = StatePredictor()

base_metrics = {
    'mean_distance': 2.1,
    'median_player1_x': 2.3,
    'median_player1_y': 6.1,
    'median_player2_x': 4.1,
    'median_player2_y': 5.9
}

prediction = predictor.predict_single(base_metrics)
print(f"Predicted state: {prediction}")
```

### Batch Prediction

```python
import pandas as pd
from predictor import StatePredictor

predictor = StatePredictor()

# Load CSV with base metrics
df = pd.read_csv("new_video_metrics.csv")
df_pred = predictor.predict_batch(df)
print(df_pred[['frame_number', 'predicted_state']])
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
    "data_path": "data/annotations",
    "model_path": "models/rally_state_model.pkl",
    "model_type": "xgboost",  # or "random_forest"
    "window_size": 50,
    "lookback_frames": 3,
    "test_size": 0.2,
    # ... other settings
}
```

## File Structure

```
├── config.py              # Configuration
├── feature_engineer.py    # Feature engineering logic
├── model_trainer.py       # Training pipeline
├── predictor.py           # Inference pipeline
├── evaluator.py          # Evaluation pipeline
├── example_usage.py      # Usage examples
└── data/
    └── annotations/      # CSV files with base metrics
```

## Key Benefits

-   **Single Source of Truth**: All pipelines use identical feature engineering
-   **Consistent Results**: Training, evaluation, and inference use the same logic
-   **No Temporal Inconsistencies**: Unified handling of lag and rolling features
-   **Modular Design**: Easy to debug, test, and maintain
-   **Configurable**: Easily switch between model types and parameters

## Solving the Original Issue

This pipeline fixes the 92% vs 68% accuracy inconsistency by:

1. **Unified Feature Engineering**: Same `FeatureEngineer` class used everywhere
2. **Consistent Temporal Processing**: Same lag and rolling feature computation
3. **No State Drift**: Removed problematic `prev_state` and `state_duration` features
4. **Sequential Processing**: Batch and single predictions use identical logic

The evaluation accuracy should now match the training test accuracy when using the same video.
