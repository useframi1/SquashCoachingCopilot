# Stroke Detection Module

The stroke detection module identifies player stroke types (forehand, backhand) in squash videos using LSTM-based analysis of player pose keypoints.

## Overview

This module provides automated stroke detection for squash videos by:

-   **LSTM-based Classification**: Uses recurrent neural network to classify stroke types
-   **Pose Keypoint Analysis**: Processes player body keypoints (COCO 17-point format)
-   **Windowed Approach**: Analyzes temporal sequences of keypoints around racket hits
-   **Normalization**: Keypoint normalization relative to body proportions
-   **Multi-player Support**: Independent predictions for both players

The module is part of the SquashCoachingCopilot package and provides essential data for shot analysis and player technique evaluation.

## Features

- **LSTM neural network** (LSTMStrokeClassifier) for temporal sequence modeling
- **Windowed prediction** around racket hit frames (31-frame sequences: ±15 frames)
- **Automatic keypoint normalization** (relative to torso length)
- **Multi-player tracking** with player-specific predictions
- **Confidence scores** for each stroke detection
- **Training utilities** for custom datasets
- **Evaluation tools** with ground truth comparison

## Components

### StrokeDetector ([stroke_detector.py](stroke_detector.py))

The main class for stroke detection following the standard module pattern.

**Key Methods:**
- `detect(input: StrokeDetectionInput) -> StrokeDetectionResult`: Main detection method
- `detect_from_dataframe(df: pd.DataFrame) -> StrokeDetectionResult`: Convenience method for evaluation

**Processing Pipeline:**
1. **Get Window**: For each racket hit, extract ±15 frames (31 total frames)
2. **Get Player Keypoints**: Use racket_hit_player_id to select the correct player's keypoints
3. **Normalize Keypoints**: Normalize relative to hip center and torso length
4. **Run Inference**: Predict stroke type using LSTMStrokeClassifier model
5. **Return Results**: Create StrokeResult for each racket hit

### Training ([train_model.py](train_model.py))

Training script following the rally state detection pattern.

**Features:**
- Loads annotated stroke sequences from CSV files
- Normalizes keypoints and creates training batches
- Trains LSTM model with validation
- Saves best model checkpoint
- Computes evaluation metrics

**Usage:**
```bash
cd squashcopilot/modules/stroke_detection
python train_model.py
```

### Evaluation ([tests/evaluator.py](tests/evaluator.py))

Comprehensive evaluation system following the standard pattern.

**Features:**
- Loads test annotations from annotation module
- Runs stroke detection on test videos
- Computes accuracy, precision, recall, F1
- Generates confusion matrix
- Creates annotated videos with predictions
- Saves results to CSV

**Usage:**
```bash
cd squashcopilot/modules/stroke_detection/tests
python evaluator.py
```

### LSTMStrokeClassifier ([model/lstm_classifier.py](model/lstm_classifier.py))

LSTM model architecture for stroke classification.

**Architecture:**
- Input: Sequential normalized keypoints (31 frames × 24 features)
  - 31 frames: ±15 frames around racket hit
  - 24 features: 12 keypoints × 2 coordinates (x, y)
- LSTM layers: 2 layers with 128 hidden units each
- Dropout: 0.3 for regularization (applied between LSTM layers and before FC layer)
- Output: 2-class classification (FOREHAND, BACKHAND)

**Implementation Details:**
- Explicitly initializes hidden states (h0) and cell states (c0) to zeros
- Uses the last time step output (`out[:, -1, :]`) for classification
- Batch-first format for efficient processing

## Data Models

The module uses standardized data models from [squashcopilot.common.models.stroke](../../common/models/stroke.py):

### Input Models
- **StrokeDetectionInput**: Input data for batch stroke detection
  - `player_keypoints`: Dict mapping player IDs to keypoint arrays (num_frames, num_keypoints, 2)
  - `racket_hits`: List of racket hit frame numbers
  - `racket_hit_player_ids`: List of player IDs for each racket hit
  - `frame_numbers`: List of all frame numbers

### Output Models
- **StrokeResult**: Stroke detection for a single racket hit
  - `frame`: Frame number of racket hit
  - `player_id`: Player identifier (1 or 2)
  - `stroke_type`: StrokeType (FOREHAND, BACKHAND)
  - `confidence`: Detection confidence (0-1)

- **StrokeDetectionResult**: Collection of all detected strokes
  - `strokes`: List of StrokeResult objects
  - Methods: `get_stroke_at_frame()`, `get_strokes_for_player()`, `get_valid_strokes()`
  - Properties: `total_strokes`, `stroke_count_by_type`, `stroke_count_by_player`

## Stroke Types

From `StrokeType` enum:
- **FOREHAND**: Forehand stroke
- **BACKHAND**: Backhand stroke
- **NEITHER**: No valid stroke (used internally)

## Usage

### Basic Detection

```python
from squashcopilot.modules.stroke_detection import StrokeDetector
import pandas as pd

# Initialize detector
detector = StrokeDetector()

# Load annotations
df = pd.read_csv("video_annotations.csv")

# Run detection
result = detector.detect_from_dataframe(df, video_name="video-1")

# Access results
for stroke in result.strokes:
    print(f"Frame {stroke.frame}: Player {stroke.player_id} - {stroke.stroke_type}")
    print(f"  Confidence: {stroke.confidence:.3f}")

# Get statistics
print(f"\nTotal strokes: {result.total_strokes}")
print(f"By type: {result.stroke_count_by_type}")
print(f"By player: {result.stroke_count_by_player}")
```

### Using Standard Data Models

```python
from squashcopilot.modules.stroke_detection import StrokeDetector
from squashcopilot.common.models.stroke import StrokeDetectionInput
import numpy as np

# Initialize detector
detector = StrokeDetector()

# Prepare input data
player_keypoints = {
    1: np.random.rand(1000, 12, 2),  # Player 1 keypoints (1000 frames, 12 keypoints, xy)
    2: np.random.rand(1000, 12, 2),  # Player 2 keypoints
}
racket_hits = [100, 250, 400, 550]  # Racket hit frame numbers
racket_hit_player_ids = [1, 2, 1, 2]  # Which player hit
frame_numbers = list(range(1000))

# Create input
input_data = StrokeDetectionInput(
    player_keypoints=player_keypoints,
    racket_hits=racket_hits,
    racket_hit_player_ids=racket_hit_player_ids,
    frame_numbers=frame_numbers,
)

# Detect strokes
result = detector.detect(input_data)

# Process results
for stroke in result.strokes:
    print(f"Detected {stroke.stroke_type} at frame {stroke.frame}")
```

### Evaluation

```python
from squashcopilot.modules.stroke_detection.tests.evaluator import StrokeDetectionEvaluator

# Initialize evaluator
evaluator = StrokeDetectionEvaluator(config_name="stroke_detection")

# Run complete evaluation pipeline
evaluator.run()

# Output includes:
# - Predictions CSV
# - Metrics (accuracy, precision, recall, F1)
# - Confusion matrix plot
# - Annotated video with predictions
```

## Configuration

Configuration file: [squashcopilot/configs/stroke_detection.yaml](../../configs/stroke_detection.yaml)

```yaml
# Inference Configuration
inference:
    method: lstm
    model_path: squashcopilot/modules/stroke_detection/model/weights/lstm_model.pt

# Normalization Settings
normalization:
    method: hip_torso
    min_torso_length: 1.0e-6

# Evaluator Configuration
evaluator:
    video_name: video-1
    data_dir: squashcopilot/modules/stroke_detection/tests/data  # Test annotations
    output_dir: squashcopilot/modules/stroke_detection/tests/outputs
    ground_truth_path: null  # Optional path to ground truth labels
    create_video: true  # Videos loaded from annotation module for visualization

# Training Configuration
training:
    data_dir: squashcopilot/modules/stroke_detection/stroke_annotations  # Windowed training sequences
    model_save_dir: squashcopilot/modules/stroke_detection/model/weights
    sequence_length: 31  # Total sequence length: ±15 frames + 1
    num_features: 24  # 12 keypoints × 2 coordinates
    batch_size: 32
    epochs: 100
    learning_rate: 0.001
    model:
        hidden_size: 128
        num_layers: 2
        dropout: 0.3
```

### Configuration Parameters

**Inference:**
- `model_path`: Path to trained LSTM model weights

**Normalization:**
- `method`: Normalization method (hip_torso)
- `min_torso_length`: Minimum torso length to avoid division by zero

**Evaluator:**
- `video_name`: Video to evaluate on
- `data_dir`: Directory with test annotations (tests/data/)
- `output_dir`: Where to save evaluation results (tests/outputs/)
- `ground_truth_path`: Optional path to ground truth stroke labels CSV
- `create_video`: Whether to generate annotated video (loads video from annotation module)

**Training:**
- `data_dir`: Directory with windowed stroke training sequences (stroke_annotations/)
- `model_save_dir`: Where to save trained models
- `sequence_length`: Total sequence length (31 frames: ±15 frames + 1)
- `num_features`: Number of input features (24: 12 keypoints × 2 coordinates)
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `learning_rate`: Optimizer learning rate
- `model.hidden_size`: LSTM hidden dimension (128)
- `model.num_layers`: Number of LSTM layers (2)
- `model.dropout`: Dropout rate for regularization (0.3)

## Algorithm Details

### Keypoint Normalization
1. **Hip Center Calculation**: Average of left and right hip positions
2. **Shoulder Center Calculation**: Average of left and right shoulder positions
3. **Torso Length Calculation**: Distance between hip center and shoulder center
4. **Normalization**: Translate keypoints relative to hip center, scale by torso length
5. **Result**: Scale and position-invariant representation

### Windowed Prediction
1. **Window Extraction**: For each racket hit at frame F, extract frames [F-15, F+15] (31 frames total)
2. **Player Selection**: Use racket_hit_player_id to get the correct player's keypoints
3. **Normalization**: Normalize keypoints relative to hip center and torso length
4. **LSTM Forward Pass**: Process 31-frame sequence through LSTMStrokeClassifier
5. **Classification**: Softmax output over classes (forehand, backhand)
6. **Result Creation**: Return StrokeResult with prediction and confidence score

### Training Data Format

Training data in `stroke_annotations/` directory:

```csv
hit_frame,frame,player_id,stroke_type,kp_left_shoulder_x,kp_left_shoulder_y,...
251,236.0,2,backhand,535.43,427.32,...
251,237.0,2,backhand,536.21,428.15,...
...
```

Each sequence:
- Grouped by `hit_frame` (the racket hit frame)
- Contains `sequence_length` frames around the hit
- Has player keypoints for the player who made the stroke
- Labeled with `stroke_type` (forehand/backhand)

## Module Structure

```
stroke_detection/
├── __init__.py                  # Package exports
├── README.md                    # This file
├── stroke_detector.py           # Main StrokeDetector class
├── train_model.py               # Training script
├── stroke_annotator.py          # Manual annotation tool
├── model/                       # Model implementations
│   ├── __init__.py
│   ├── lstm_classifier.py       # LSTM architecture
│   └── weights/                 # Model checkpoints
│       └── lstm_model.pt        # Trained model
├── stroke_annotations/          # Training data (windowed & labeled sequences)
│   ├── video-1_strokes_annotated.csv
│   ├── video-2_strokes_annotated.csv
│   └── ...
└── tests/                       # Testing and evaluation
    ├── evaluator.py             # Evaluation script
    ├── data/                    # Test annotations (symlink or copy from annotation module)
    │   ├── video-1/
    │   │   └── video-1_annotations.csv
    │   └── video-2/
    │       └── video-2_annotations.csv
    └── outputs/                 # Evaluation results
        └── video-1/
            ├── video-1_predictions.csv
            ├── video-1_confusion_matrix.png
            ├── video-1_metrics.txt
            └── video-1_strokes_annotated.mp4
```

**Note on Data Directories:**
- `stroke_annotations/`: Contains windowed sequences with stroke labels (for training)
- `tests/data/`: Contains raw annotation CSVs from the annotation module (for evaluation)
- Videos are loaded from `squashcopilot/annotation/annotations/{video_name}/` when creating annotated output videos

## Testing and Evaluation

### Running Evaluation

```bash
# Configure evaluator in stroke_detection.yaml
# Set video_name, data_dir, output_dir, etc.

# Run evaluator
cd squashcopilot/modules/stroke_detection/tests
python evaluator.py
```

### Evaluation Output

The evaluator generates:
1. **Predictions CSV**: Frame-by-frame predictions with confidence scores
2. **Metrics File**: Accuracy, precision, recall, F1 (overall and per-class)
3. **Confusion Matrix**: Visualization of classification performance
4. **Annotated Video**: Video with ground truth and predictions overlaid

### Ground Truth Labels

Ground truth can be provided in two ways:
1. **Separate CSV**: Set `ground_truth_path` in config pointing to CSV with columns: `frame`, `stroke_type`
2. **In Annotations**: Add `stroke_type` column to the annotations CSV

## Training Custom Models

```bash
# 1. Prepare training data using stroke_annotator.py
python stroke_annotator.py

# 2. Configure training in stroke_detection.yaml
# Set data_dir, epochs, batch_size, etc.

# 3. Run training
python train_model.py

# 4. Model will be saved to model/weights/stroke_lstm_best.pt
```

## Performance Considerations

- **LSTM Inference**: ~500-1000 strokes/second on GPU, ~100-200/second on CPU
- **Keypoint Normalization**: Real-time (>10000 FPS)
- **Memory**: Minimal (<50MB for model)
- **Sequence Length**: 31 frames (±15 frames around racket hit) provides good balance of context and efficiency

## Limitations

- Requires accurate pose keypoints from player tracking
- Requires racket hit detection (is_racket_hit, racket_hit_player_id in annotations)
- Model trained on specific dataset may not generalize universally
- Cannot distinguish between different stroke techniques (drive, drop, volley, etc.)
- Assumes standard squash stroke biomechanics
- Window must be fully contained within video bounds

## Dependencies on Other Modules

This module requires data from:
- **Player Tracking**: Player pose keypoints (COCO format)
- **Ball Tracking**: Racket hit detection (is_racket_hit, racket_hit_player_id)

Provides data to:
- **Shot Classification**: Can be combined with shot analysis
- **Performance Analysis**: Stroke statistics and patterns

## Similar Modules

This module follows the same structure as:
- **Rally State Detection**: See [rally_state_detection/README.md](../rally_state_detection/README.md)
- **Shot Type Classification**: See [shot_type_classification/README.md](../shot_type_classification/README.md)

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

MIT License

This module is part of the SquashCoachingCopilot thesis project at the American University in Cairo.
