# Rally State Detection Module

The rally state detection module segments squash videos into individual rallies using LSTM-based temporal analysis of ball and player trajectories.

## Overview

This module provides automated rally segmentation for squash videos by:

-   **LSTM-based Detection**: Uses recurrent neural network to detect rally START and END states
-   **Multi-feature Input**: Analyzes ball Y-position and player positions in meters
-   **Temporal Modeling**: Learns temporal patterns of rally transitions
-   **Post-processing**: Merges short segments and applies minimum rally length constraints
-   **Training Support**: Includes trainer for custom datasets

The module is part of the SquashCoachingCopilot package and enables rally-level analysis and statistics.

## Features

- **LSTM neural network** for temporal sequence modeling
- **Multi-feature input** (ball Y + player X/Y positions)
- **Bidirectional LSTM** option for context-aware predictions
- **Configurable architecture** (layers, hidden size, dropout)
- **Post-processing pipeline** for clean rally segments
- **Training utilities** for custom datasets
- **Rally statistics** (duration, count, average length)

## Components

### RallyStateDetector (`rally_state_detector.py`)

The main class for rally segmentation.

**Key Methods:**
- `detect(input: RallySegmentationInput)`: Frame-by-frame LSTM inference
- `segment_video(input: RallySegmentationInput)`: Generate rally segments
- `reset()`: Reset detector state

**Processing Pipeline:**
1. **Feature Extraction**: Collect ball Y and player positions per frame
2. **Normalization**: Normalize features to [0, 1] range
3. **LSTM Inference**: Predict rally state (START/END) per frame
4. **Post-processing**:
   - Minimum segment length filtering (120 frames)
   - Gap merging (merge segments separated by <60 frames)
   - Rally ID assignment

### RallyStateLSTM (`models/rally_state_lstm.py`)

LSTM model for rally state prediction.

**Architecture:**
- Input: Sequential features (ball_y, player1_x, player1_y, player2_x, player2_y)
- LSTM layers: 2 layers, 128 hidden units each
- Dropout: 0.3 for regularization
- Bidirectional: Optional (default: True)
- Output: Binary classification (START vs END)

## Data Models

The module uses standardized data models from `squashcopilot.common.models.rally`:

### Input Models
- **RallySegmentationInput**: Trajectory data for rally detection
  - `ball_y_positions`: List of ball Y-coordinates (meters) per frame
  - `player_positions`: Optional dict of player positions (meters) per frame
  - `fps`: Video frame rate for duration calculation

### Output Models
- **RallySegment**: Single rally segment
  - `rally_id`: Unique identifier (1, 2, 3, ...)
  - `start_frame`: First frame of rally
  - `end_frame`: Last frame of rally
  - `duration_seconds`: Rally length in seconds
  - `frame_count`: Number of frames in rally

- **RallySegmentationResult**: Complete segmentation results
  - `segments`: List of RallySegment objects
  - `total_rallies`: Total number of rallies detected
  - `average_duration`: Mean rally duration in seconds
  - `total_duration`: Sum of all rally durations

## Usage

### Using Standard Data Models

```python
from squashcopilot import RallyStateDetector
from squashcopilot import RallySegmentationInput

# Initialize detector
detector = RallyStateDetector()

# Prepare input (ball Y positions + player positions)
input_data = RallySegmentationInput(
    ball_y_positions=ball_trajectory_y_meters,  # List[float]
    player_positions=player_positions_meters,    # Dict[frame] -> Dict[player_id] -> Point2D
    fps=30.0
)

# Segment video into rallies
result = detector.segment_video(input_data)

# Access rally segments
for rally in result.segments:
    print(f"Rally {rally.rally_id}:")
    print(f"  Frames: {rally.start_frame} to {rally.end_frame}")
    print(f"  Duration: {rally.duration_seconds:.2f}s")
    print(f"  Frame count: {rally.frame_count}")

# Get statistics
print(f"Total rallies: {result.total_rallies}")
print(f"Average duration: {result.average_duration:.2f}s")
```

### Integration with Other Modules

```python
from squashcopilot import Annotator

# Annotator handles rally segmentation automatically (if enabled)
annotator = Annotator()
results = annotator.annotate_video("video.mp4", "output_directory")

# Access rally segments
rally_result = results['rally_segments']
```

## Configuration

Configuration file: `squashcopilot/configs/rally_state_detection.yaml`

```yaml
model:
  path: "models/rally_lstm_best.pth"
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true

features:
  use_ball_y: true
  use_player_positions: true  # Requires player tracking

inference:
  device: "auto"  # auto, cuda, cpu
  batch_size: 32

post_processing:
  min_segment_length: 120  # Minimum rally length in frames
  gap_merge_threshold: 60  # Merge segments separated by less than this

annotation:
  data_dir: "annotation/data"
  output_dir: "annotation/annotations"
```

### Configuration Parameters

**Model:**
- `path`: Path to trained LSTM model weights
- `hidden_size`: LSTM hidden state dimension (128)
- `num_layers`: Number of LSTM layers (2)
- `dropout`: Dropout rate for regularization (0.3)
- `bidirectional`: Use bidirectional LSTM (true)

**Features:**
- `use_ball_y`: Include ball Y-position as feature (true)
- `use_player_positions`: Include player X/Y positions (true, requires player tracking)

**Inference:**
- `device`: Computation device (auto/cuda/cpu)
- `batch_size`: Batch size for inference (32)

**Post-processing:**
- `min_segment_length`: Minimum rally length in frames (120 = 4s at 30fps)
- `gap_merge_threshold`: Merge rallies separated by <60 frames (2s)

## Algorithm Details

### LSTM Prediction
1. **Input Features**: For each frame t:
   - Ball Y-position (meters, normalized)
   - Player 1 X, Y positions (meters, normalized)
   - Player 2 X, Y positions (meters, normalized)
2. **LSTM Forward Pass**: Process sequential features
3. **Classification**: Sigmoid output → probability of rally state
4. **Thresholding**: prob > 0.5 → START, else → END

### Post-processing Pipeline
1. **Initial Segmentation**: Group consecutive START frames
2. **Minimum Length Filter**: Remove rallies < 120 frames
3. **Gap Merging**: Merge rallies separated by < 60 frames
4. **Rally ID Assignment**: Assign sequential IDs (1, 2, 3, ...)
5. **Duration Calculation**: Convert frame counts to seconds using FPS

### Training (Optional)

The module includes `train_model.py` for training custom models:

```python
from squashcopilot.modules.rally_state_detection.train_model import train_rally_lstm

# Prepare training data
train_data = {
    'features': [...],  # Shape: (num_samples, sequence_length, num_features)
    'labels': [...]     # Shape: (num_samples, sequence_length)
}

# Train model
model = train_rally_lstm(
    train_data=train_data,
    val_data=val_data,
    epochs=50,
    learning_rate=0.001
)
```

## Module Structure

```
rally_state_detection/
├── __init__.py                  # Package exports
├── rally_state_detector.py      # Main RallyStateDetector class
├── models/
│   ├── __init__.py
│   └── rally_state_lstm.py      # LSTM model architecture
├── train_model.py               # Training utilities
├── annotator.py                 # Data annotation tool
└── tests/                       # Evaluation suite
    ├── evaluator.py             # Evaluation script
    ├── data/                    # Test videos
    └── outputs/                 # Results and metrics
```

## Testing and Evaluation

The `tests/` directory contains evaluation tools:

- **evaluator.py**: Tests segmentation accuracy against manual annotations
- **data/**: Test videos with ground truth rally boundaries
- **outputs/**: Segmentation results and metrics

### Running Tests

```python
from squashcopilot.modules.rally_state_detection.tests.evaluator import RallyStateEvaluator

evaluator = RallyStateEvaluator()
results = evaluator.evaluate(video_path="tests/data/video-1.mp4")
evaluator.generate_report(results)
```

## Performance Considerations

- **LSTM Inference**: ~500-1000 FPS on GPU, ~100-200 FPS on CPU
- **Post-processing**: Real-time (>5000 FPS)
- **Memory**: Minimal (<100MB for model)
- **Latency**: Batch processing reduces inference time

## Limitations

- Requires accurate ball tracking (ball Y-positions)
- Model trained on specific dataset may not generalize to all videos
- Short rallies (<4s) are filtered out by default
- Performance degrades with poor ball tracking quality
- May miss rally boundaries during very short inter-rally periods

## Dependencies on Other Modules

This module requires data from:
- **Ball Tracking**: Ball Y-positions in meters
- **Player Tracking** (optional): Player positions for improved accuracy
- **Court Calibration**: Homography for meter coordinate conversion

## Training Data

The model is trained on annotated squash videos with manual rally boundary labels. To create custom training data:

1. Use `annotator.py` to label rally boundaries
2. Export features (ball Y + player positions)
3. Train using `train_model.py`

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

MIT License

This module is part of the SquashCoachingCopilot thesis project at the American University in Cairo.
