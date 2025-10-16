# Ball Detection Pipeline

A Python package for real-time ball detection and tracking in squash videos with support for multiple detection models.

## Overview

This package provides a unified `BallTracker` interface that supports multiple ball detection implementations:
- **TrackNet**: A deep learning CNN model that processes sequences of three consecutive frames for accurate ball position prediction
- **Roboflow (RF)**: Cloud-based inference using pre-trained models from the Roboflow platform

The tracker automatically handles frame buffering, coordinate scaling, and model selection based on configuration.

## Features

- Multiple tracker support (TrackNet and Roboflow)
- Real-time ball detection and tracking
- GPU acceleration support (CUDA) for TrackNet
- Automatic coordinate scaling to match original frame resolution
- Configurable model parameters via JSON configuration
- Frame buffering for temporal context (TrackNet)
- Simple, unified API for all tracker types

## Installation

### Basic Installation (TrackNet only)

```bash
pip install -e .
```

### With Roboflow Support

```bash
pip install -e ".[roboflow]"
```

### Full Installation (All Features)

```bash
pip install -e ".[all]"
```

### Requirements

**Core Dependencies:**
- Python >= 3.7
- PyTorch
- OpenCV
- NumPy

**Optional Dependencies (for Roboflow tracker):**
- `inference` - Roboflow inference SDK
- `supervision` - Detection result processing

## Usage

### Basic Example

```python
from ball_detection_pipeline import BallTracker
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

### Custom Configuration

```python
# Load custom configuration
custom_config = {
    "tracker": {
        "type": "tracknet"  # or "rf" for Roboflow
    },
    "tracknet_model": {
        "model_path": "models/tracknet/weights/model_best.pt",
        "device": "cuda",  # "cuda", "cpu", or "auto"
        "model_width": 640,
        "model_height": 360
    }
}

tracker = BallTracker(config=custom_config)
```

### Reset Tracker State

```python
# Reset the frame buffer (useful when switching videos)
tracker.reset()
```

## How It Works

### TrackNet Tracker
1. **Frame Buffering**: Maintains a buffer of the last 3 frames
2. **Model Input**: Three consecutive frames are concatenated (9 channels)
3. **Inference**: BallTrackerNet CNN outputs a heatmap prediction
4. **Post-processing**: Heatmap is converted to (x, y) coordinates using Hough circle detection
5. **Coordinate Scaling**: Coordinates are scaled from model resolution (640×360) to original frame resolution

### Roboflow Tracker
1. **API Inference**: Frame is sent to Roboflow inference API
2. **Detection**: Pre-trained model detects ball with bounding box
3. **Center Extraction**: Highest confidence detection center is calculated
4. **Return**: (x, y) coordinates in original frame resolution

## Model Architectures

### TrackNet (BallTrackerNet)
- **Input**: 9 channels (3 consecutive RGB frames)
- **Resolution**: 640×360
- **Architecture**: Encoder-decoder U-Net style CNN
- **Output**: 256-channel heatmap for ball position
- **Features**: Skip connections, batch normalization, ReLU activation
- **Training**: Trained on squash video datasets

### Roboflow
- **Model**: Pre-trained on Roboflow platform
- **Type**: Object detection (bounding box prediction)
- **Inference**: Cloud-based API
- **Model ID**: `squash-ball-detection-1lbti/1`

## Configuration

Default configuration ([config.json](ball_detection_pipeline/config.json)):

```json
{
    "tracker": {
        "type": "rf"
    },
    "tracknet_model": {
        "model_path": "models/tracknet/weights/model_best.pt",
        "device": "auto",
        "model_width": 640,
        "model_height": 360
    },
    "rf_model": {
        "model_id": "squash-ball-detection-1lbti/1",
        "api_key": "YOUR_API_KEY"
    }
}
```

### Configuration Parameters

- **tracker.type**: Select tracker implementation (`"tracknet"` or `"rf"`)
- **tracknet_model.model_path**: Path to TrackNet model weights
- **tracknet_model.device**: Device for inference (`"auto"`, `"cuda"`, or `"cpu"`)
- **tracknet_model.model_width/height**: Input resolution for TrackNet
- **rf_model.model_id**: Roboflow model identifier
- **rf_model.api_key**: Roboflow API authentication key

## API Reference

### BallTracker

#### `__init__(config: dict = None)`

Initialize the ball tracker with optional configuration.

**Parameters:**
- `config` (dict, optional): Configuration dictionary. If None, loads from `config.json`

**Raises:**
- `ValueError`: If tracker type is invalid

#### `process_frame(frame) -> tuple`

Process a single frame and return ball coordinates.

**Parameters:**
- `frame` (numpy.ndarray): Input frame in BGR format (any resolution)

**Returns:**
- `tuple`: (x, y) coordinates of the ball in original frame resolution, or (None, None) if not detected

#### `reset()`

Reset the tracker state by clearing the frame buffer (TrackNet only).

## Package Structure

```
ball_detection_pipeline/
├── __init__.py                 # Package initialization (exports BallTracker)
├── ball_tracker.py             # Main BallTracker wrapper class
├── utils.py                    # Utility functions (config loading, postprocessing)
├── config.json                 # Default configuration
└── models/                     # Model implementations
    ├── __init__.py
    ├── tracknet/               # TrackNet-based tracker
    │   ├── __init__.py
    │   ├── model.py            # BallTrackerNet CNN architecture
    │   ├── tracknet_tracker.py # TrackNetTracker implementation
    │   └── weights/            # Pre-trained model weights
    │       └── model_best.pt
    └── rf/                     # Roboflow-based tracker
        ├── __init__.py
        └── rf_tracker.py       # RFTracker implementation
```

## Choosing a Tracker

### TrackNet
**Pros:**
- Local inference (no internet required)
- Temporal context from 3-frame sequences
- GPU acceleration available
- No API rate limits

**Cons:**
- Requires model weights file
- Higher computational requirements
- GPU recommended for real-time performance

### Roboflow
**Pros:**
- No local model required
- Fast cloud-based inference
- Easy model updates via platform
- Lower local computational requirements

**Cons:**
- Requires internet connection
- API key required
- Potential rate limits
- Latency from API calls

## Development

### Package Version
Current version: **0.1.5**

### Recent Updates
- Added Roboflow tracker support
- Restructured package with multiple model implementations
- Unified BallTracker interface
- Updated configuration system
- Improved coordinate scaling and postprocessing

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

This project is part of a thesis on Squash Coaching Copilot at the American University in Cairo.
