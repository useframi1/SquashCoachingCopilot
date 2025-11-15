# Ball Tracking

A Python package for real-time ball detection and tracking in squash videos using the TrackNet deep learning model.

## Overview

This package provides ball tracking capabilities using TrackNet, a deep learning CNN model that processes sequences of three consecutive frames for accurate ball position prediction. It includes preprocessing, postprocessing, and hit detection features.

## Features

- **Real-time ball tracking** using TrackNet model
- **GPU acceleration** support (CUDA)
- **Preprocessing** for black ball enhancement (CLAHE, dilation)
- **Postprocessing** with outlier removal and interpolation
- **Wall hit detection** using local minima in ball trajectory
- **Racket hit detection** using slope analysis
- Automatic coordinate scaling to match original frame resolution
- Frame buffering for temporal context
- Configurable via JSON configuration

## Installation

### Basic Installation

```bash
pip install -e .
```

### Requirements

**Core Dependencies:**
- Python >= 3.7
- PyTorch
- OpenCV
- NumPy
- SciPy

## Usage

### Basic Example

```python
from ball_tracking import BallTracker
import cv2

# Initialize the tracker (uses config.json by default)
tracker = BallTracker()

# Process video frames
cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get ball coordinates
    x, y = tracker.process_frame(frame)

    if x is not None and y is not None:
        # Draw ball position
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Ball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### With Preprocessing and Postprocessing

```python
from ball_tracking import BallTracker

tracker = BallTracker()

# Collect positions for all frames
positions = []
for frame in video_frames:
    # Preprocessing is done internally if is_black_ball=True in config
    x, y = tracker.process_frame(frame)
    positions.append((x, y))

# Apply postprocessing
smoothed_positions = tracker.postprocess(positions)
```

### Hit Detection

```python
from ball_tracking import BallTracker, WallHitDetector, RacketHitDetector

tracker = BallTracker()
wall_detector = WallHitDetector()
racket_detector = RacketHitDetector()

# Track ball positions
positions = []
for frame in video_frames:
    x, y = tracker.process_frame(frame)
    positions.append((x, y))

# Apply postprocessing
positions = tracker.postprocess(positions)

# Detect hits
wall_hits = wall_detector.detect(positions)
racket_hits = racket_detector.detect(positions, wall_hits)

print(f"Detected {len(wall_hits)} wall hits")
print(f"Detected {len(racket_hits)} racket hits")
```

### Reset Tracker State

```python
# Reset the frame buffer (useful when switching videos)
tracker.reset()
```

## How It Works

### TrackNet Tracker
1. **Frame Buffering**: Maintains a buffer of the last 3 frames
2. **Preprocessing** (optional): Applies CLAHE and dilation for black ball enhancement
3. **Model Input**: Three consecutive frames are concatenated (9 channels)
4. **Inference**: BallTrackerNet CNN outputs a heatmap prediction
5. **Post-processing**: Heatmap is converted to (x, y) coordinates using Hough circle detection
6. **Coordinate Scaling**: Coordinates are scaled from model resolution (640×360) to original frame resolution

### Postprocessing Pipeline
1. **Outlier Removal**: Removes positions that are too far from neighboring positions
2. **Interpolation**: Fills missing values using linear interpolation

### Wall Hit Detection
- Finds local minima in Y-coordinate trajectory (ball closest to front wall)
- Validates hits based on prominence, width, and minimum distance

### Racket Hit Detection
- Analyzes steep negative slopes in Y-coordinate before wall hits
- Identifies points where ball accelerates toward the wall

## Model Architecture

### TrackNet (BallTrackerNet)
- **Input**: 9 channels (3 consecutive RGB frames)
- **Resolution**: 640×360
- **Architecture**: Encoder-decoder U-Net style CNN
- **Output**: 256-channel heatmap for ball position
- **Features**: Skip connections, batch normalization, ReLU activation
- **Training**: Trained on squash video datasets

## Configuration

Default configuration ([ball_tracking/config.json](ball_tracking/config.json)):

```json
{
    "tracker": {
        "is_black_ball": false
    },
    "model": {
        "model_path": "models/tracknet/weights/model_best.pt",
        "device": "auto",
        "model_width": 640,
        "model_height": 360
    },
    "postprocessing": {
        "enabled": true,
        "outlier_detection": {
            "enabled": true,
            "window": 20,
            "threshold": 80
        }
    },
    "wall_hit_detection": {
        "prominence": 50.0,
        "width": 10,
        "min_distance": 50
    },
    "racket_hit_detection": {
        "slope_window": 10,
        "slope_threshold": 3.0,
        "min_distance": 15,
        "lookback_frames": 50
    }
}
```

### Configuration Parameters

**Tracker:**
- `is_black_ball`: Enable preprocessing for black ball detection

**Model:**
- `model_path`: Path to TrackNet model weights
- `device`: Device for inference (`"auto"`, `"cuda"`, or `"cpu"`)
- `model_width/height`: Input resolution for TrackNet

**Postprocessing:**
- `enabled`: Enable postprocessing pipeline
- `outlier_detection.window`: Rolling window size for outlier detection
- `outlier_detection.threshold`: Distance threshold in pixels

**Wall Hit Detection:**
- `prominence`: Minimum depth of valley in pixels
- `width`: Minimum width of valley in frames
- `min_distance`: Minimum frames between consecutive hits

**Racket Hit Detection:**
- `slope_window`: Number of frames to calculate slope over
- `slope_threshold`: Minimum absolute slope (pixels/frame)
- `min_distance`: Minimum frames between consecutive hits
- `lookback_frames`: How many frames to look back from wall hit

## API Reference

### BallTracker

#### `__init__(config: dict = None)`
Initialize the ball tracker with optional configuration.

**Parameters:**
- `config` (dict, optional): Configuration dictionary. If None, loads from `config.json`

#### `process_frame(frame) -> tuple`
Process a single frame and return ball coordinates.

**Parameters:**
- `frame` (numpy.ndarray): Input frame in BGR format (any resolution)

**Returns:**
- `tuple`: (x, y) coordinates of the ball, or (None, None) if not detected

#### `preprocess_frame(frame) -> ndarray`
Preprocess frame for black ball detection (CLAHE + dilation).

**Parameters:**
- `frame` (numpy.ndarray): Input frame in BGR format

**Returns:**
- `numpy.ndarray`: Preprocessed frame

#### `postprocess(positions, config=None) -> list`
Apply postprocessing pipeline to ball positions.

**Parameters:**
- `positions` (list): List of (x, y) tuples
- `config` (dict, optional): Postprocessing configuration

**Returns:**
- `list`: Smoothed positions with outliers removed

#### `reset()`
Reset the tracker state by clearing the frame buffer.

### WallHitDetector

#### `__init__(config: dict = None)`
Initialize the wall hit detector.

#### `detect(positions) -> list`
Detect front wall hits from ball trajectory.

**Parameters:**
- `positions` (list): List of (x, y) tuples

**Returns:**
- `list`: List of hit dictionaries with frame, x, y, prominence

### RacketHitDetector

#### `__init__(config: dict = None)`
Initialize the racket hit detector.

#### `detect(positions, wall_hits) -> list`
Detect racket hits from ball trajectory.

**Parameters:**
- `positions` (list): List of (x, y) tuples
- `wall_hits` (list): List of wall hit dictionaries

**Returns:**
- `list`: List of hit dictionaries with frame, x, y, slope

## Package Structure

```
ball_tracking/
├── __init__.py                    # Package exports
├── ball_tracker.py                # Main BallTracker class
├── wall_hit_detector.py           # WallHitDetector class
├── racket_hit_detector.py         # RacketHitDetector class
├── utils.py                       # Utility functions
├── config.json                    # Default configuration
└── models/                        # Model implementations
    ├── __init__.py
    └── tracknet/                  # TrackNet implementation
        ├── __init__.py
        ├── model.py               # BallTrackerNet architecture
        ├── tracknet_tracker.py    # TrackNetTracker class
        └── weights/               # Pre-trained weights
            └── model_best.pt

tests/                             # Evaluation suite (not distributed)
├── evaluator.py                   # Evaluation script
├── config.json                    # Test configuration
└── README.md                      # Test documentation
```

## Development

### Package Version
Current version: **0.1.0**

### Building and Installing

```bash
# Build the package
pip install build
python -m build

# Install from source
pip install .

# Development installation
pip install -e .
```

### Running Tests

```bash
cd tests
python evaluator.py
```

See [tests/README.md](tests/README.md) for more details on evaluation.

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

MIT License

This project is part of a thesis on Squash Coaching Copilot at the American University in Cairo.
