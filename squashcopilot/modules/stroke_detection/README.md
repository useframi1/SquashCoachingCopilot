# Stroke Detection Module

The stroke detection module identifies player stroke types (forehand, backhand) in squash videos using LSTM-based analysis of player pose keypoints around racket hit events.

## Overview

This module provides automated stroke detection for squash videos by:

-   **LSTM-based Classification**: Uses recurrent neural network to classify stroke types from temporal keypoint sequences
-   **Pose Keypoint Analysis**: Processes 12 player body keypoints (subset of COCO format: shoulders, elbows, wrists, hips, knees, ankles)
-   **Windowed Approach**: Analyzes 31-frame sequences (±15 frames) around each racket hit
-   **Normalization**: Hip-torso normalization for scale and position invariance
-   **Multi-player Support**: Independent predictions for both players using player-specific keypoints

The module is part of the SquashCoachingCopilot package (Stage 6) and provides stroke type information essential for comprehensive shot analysis and player technique evaluation.

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

### Keypoint Normalization (Hip-Torso Method)
The normalization process makes keypoints scale and position-invariant:

1. **Hip Center Calculation**: Average of left and right hip positions → (hip_center_x, hip_center_y)
2. **Shoulder Center Calculation**: Average of left and right shoulder positions → (shoulder_center_x, shoulder_center_y)
3. **Torso Length Calculation**: Euclidean distance between hip center and shoulder center
4. **Normalization**: For each keypoint (x, y):
   - Translate: `x' = x - hip_center_x`, `y' = y - hip_center_y`
   - Scale: `x_norm = x' / torso_length`, `y_norm = y' / torso_length`
5. **Result**: Scale and position-invariant representation centered at origin

**Note**: Minimum torso length threshold (1.0e-6) prevents division by zero in edge cases.

### Windowed Prediction Pipeline
For each detected racket hit, the module:

1. **Window Extraction**: For racket hit at frame F, extract frames [F-15, F+15] (31 frames total)
   - Validates that all 31 frames exist in the input data
   - Skips hits without sufficient surrounding frames
2. **Player Selection**: Uses `racket_hit_player_id` to select the correct player's keypoints
3. **Keypoint Extraction**: Retrieves 12 keypoints × 2 coordinates = 24 features per frame
4. **Normalization**: Applies hip-torso normalization to the 31-frame sequence
5. **LSTM Forward Pass**: Processes sequence through 2-layer LSTM (128 hidden units)
6. **Classification**: Applies softmax to get class probabilities (forehand/backhand)
7. **Result Creation**: Returns StrokeResult with predicted type and confidence score

### Training Data Format

Training data is organized in video-level subdirectories under `tests/data/`:

```
tests/data/
├── video-1/
│   └── video-1_annotations.csv
├── video-2/
│   └── video-2_annotations.csv
└── ...
```

Each annotations CSV must contain:
- **Frame metadata**: `frame`, `hit_frame` (groups 31-frame sequences)
- **Player info**: `player_id` (1 or 2)
- **Stroke label**: `stroke_type` (forehand, backhand)
- **Keypoints**: `kp_{keypoint_name}_x`, `kp_{keypoint_name}_y` for all 12 keypoints

Example CSV structure:
```csv
hit_frame,frame,player_id,stroke_type,kp_left_shoulder_x,kp_left_shoulder_y,...
251,236,2,backhand,535.43,427.32,...
251,237,2,backhand,536.21,428.15,...
251,238,2,backhand,537.05,428.90,...
...
251,266,2,backhand,545.12,435.67,...
```

**Training Data Characteristics**:
- Each `hit_frame` groups exactly 31 consecutive frames (sequence_length)
- Frames are centered on racket hit: [hit_frame-15, ..., hit_frame, ..., hit_frame+15]
- Only the player who made the stroke has keypoints in each sequence
- All frames in a sequence share the same `stroke_type` label
- Training script performs video-level splitting to prevent data leakage

## Module Structure

```
stroke_detection/
├── __init__.py                  # Package exports (StrokeDetector)
├── README.md                    # This file
├── stroke_detector.py           # Main StrokeDetector class
├── train_model.py               # Training script with StrokeTrainer
├── stroke_annotator.py          # Manual annotation tool (optional)
├── model/                       # Model implementations
│   ├── __init__.py
│   ├── lstm_classifier.py       # LSTMStrokeClassifier (2-layer LSTM)
│   └── weights/                 # Model checkpoints
│       ├── lstm_model.pt        # Trained model (used by detector)
│       └── stroke_lstm_best.pt  # Best checkpoint from training
└── tests/                       # Testing and evaluation
    ├── evaluator.py             # StrokeDetectionEvaluator
    ├── data/                    # Training/test annotations (video-level organization)
    │   ├── video-1/
    │   │   └── video-1_annotations.csv  # Contains hit_frame, frame, stroke_type, keypoints
    │   ├── video-2/
    │   │   └── video-2_annotations.csv
    │   └── ...
    └── outputs/                 # Evaluation results
        └── video-{N}/
            ├── video-{N}_predictions.csv        # Frame-by-frame predictions
            ├── video-{N}_confusion_matrix.png   # Classification performance
            ├── video-{N}_metrics.txt            # Accuracy, precision, recall, F1
            └── video-{N}_strokes_annotated.mp4  # Annotated video
```

**Data Organization**:
- `tests/data/`: Contains video-level subdirectories with annotated stroke sequences
  - Each CSV has 31-frame windows grouped by `hit_frame` with stroke labels
  - Used by both training (train_model.py) and evaluation (tests/evaluator.py)
- `model/weights/`: Stores trained LSTM model checkpoints
  - `lstm_model.pt`: Model loaded by StrokeDetector for inference
  - `stroke_lstm_best.pt`: Best model saved during training
- Training performs video-level splitting to prevent data leakage between train/val/test sets

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

### Prerequisites
Training data must be organized as video-level subdirectories in `tests/data/`, with each subdirectory containing a CSV file with 31-frame stroke sequences (see Training Data Format above).

### Training Steps

```bash
# 1. Configure training in stroke_detection.yaml
# Set data_dir, epochs, batch_size, train_test_split, validation_split, etc.

# 2. Run training script
cd squashcopilot/modules/stroke_detection
python train_model.py

# 3. Model checkpoints will be saved to model/weights/
# - stroke_lstm_best.pt (best validation accuracy)

# 4. Update inference.model_path in config to use new model
# Set model_path to: squashcopilot/modules/stroke_detection/model/weights/stroke_lstm_best.pt
```

### Training Process
The `StrokeTrainer` class handles:
1. **Data Loading**: Loads annotated sequences from all video subdirectories
2. **Video-Level Splitting**: Splits videos (not sequences) into train/val/test sets to prevent data leakage
3. **Normalization**: Applies hip-torso normalization to all keypoint sequences
4. **Training**: Trains 2-layer LSTM with cross-entropy loss and Adam optimizer
5. **Validation**: Monitors validation accuracy and saves best model
6. **Evaluation**: Computes final metrics (accuracy, precision, recall, F1) on test set

### Data Splitting Strategy
- Videos are randomly shuffled and split at the video level
- Default: 80% train+val, 20% test (configurable via `train_test_split`)
- Validation split from training videos: 10% (configurable via `validation_split`)
- This ensures no sequences from the same video appear in both train and test sets

## Performance Considerations

- **LSTM Inference**: ~500-1000 strokes/second on GPU, ~100-200/second on CPU
- **Keypoint Normalization**: Real-time (>10000 FPS)
- **Memory**: Minimal (<50MB for model)
- **Sequence Length**: 31 frames (±15 frames around racket hit) provides good balance of context and efficiency

## Limitations

- **Requires accurate pose keypoints**: Depends on player tracking module providing reliable 12-keypoint poses
- **Requires racket hit detection**: Must have `is_racket_hit` and `racket_hit_player_id` from ball tracking module
- **Fixed window size**: Requires 31 consecutive frames (±15 around hit); skips hits near video boundaries
- **Binary classification only**: Classifies forehand vs backhand, not stroke quality or specific techniques
- **Model generalization**: Performance depends on training data diversity (court types, player styles, camera angles)
- **No real-time capability**: Designed for batch processing, not live analysis
- **Player ID dependency**: Requires consistent player identification across frames from player tracking

## Dependencies on Other Modules

### Required Input (Stage 2-5 outputs)
This module requires:
- **Player Tracking** (Stage 2): Player pose keypoints for both players
  - 12 keypoints per player: shoulders, elbows, wrists, hips, knees, ankles
  - Consistent player IDs across frames
- **Ball Tracking - Racket Hit Detection** (Stage 5): Racket hit events
  - `is_racket_hit`: Boolean flag for racket hit frames
  - `racket_hit_player_id`: Which player (1 or 2) made the hit

### Provides Output To (Stage 7+)
- **Shot Classification**: Stroke type can be combined with shot direction/depth for complete shot analysis
- **Performance Analysis**: Stroke statistics per player (forehand/backhand ratios, confidence scores)
- **Tactical Analysis**: Stroke patterns and preferences across rallies

## Similar Modules

This module follows the same structure as:
- **Rally State Detection**: See [rally_state_detection/README.md](../rally_state_detection/README.md)
- **Shot Type Classification**: See [shot_type_classification/README.md](../shot_type_classification/README.md)

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

MIT License

This module is part of the SquashCoachingCopilot thesis project at the American University in Cairo.
