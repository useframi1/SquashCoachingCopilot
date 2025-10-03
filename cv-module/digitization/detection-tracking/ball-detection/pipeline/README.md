# Ball Detection Pipeline

A Python package for real-time ball detection in squash videos using a TrackNet-based deep learning model.

## Overview

This package provides a `BallTracker` class that uses a convolutional neural network to detect and track the ball in squash video frames. The model processes sequences of three consecutive frames to accurately predict the ball's position.

## Features

-   Real-time ball detection using TrackNet architecture
-   GPU acceleration support (CUDA)
-   Automatic coordinate scaling to match original frame resolution
-   Configurable model parameters via JSON configuration
-   Frame buffering for temporal context

## Installation

```bash
pip install -e .
```

### Requirements

-   Python >= 3.7
-   PyTorch
-   OpenCV
-   NumPy

## Usage

### Basic Example

```python
from ball_detection_pipeline import BallTracker
import cv2

# Initialize the tracker
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
    "model_path": "model/weights/model.pkl",
    "model_width": 640,
    "model_height": 360
}

tracker = BallTracker(config=custom_config)
```

### Reset Tracker State

```python
# Reset the frame buffer (useful when switching videos)
tracker.reset()
```

## How It Works

1. **Frame Buffering**: The tracker maintains a buffer of the last 3 frames
2. **Model Input**: Three consecutive frames are concatenated and fed to the TrackNet model
3. **Inference**: The model outputs a heatmap prediction for ball location
4. **Post-processing**: The heatmap is converted to (x, y) coordinates
5. **Coordinate Scaling**: Coordinates are scaled from model resolution to original frame resolution

## Model Architecture

The package uses a TrackNet-based architecture (`BallTrackerNet`) that:

-   Takes 3 consecutive frames as input (9 channels total)
-   Processes frames at 640x360 resolution
-   Outputs ball position predictions

## Configuration

Default configuration (`config.json`):

```json
{
    "model_path": "model/weights/model.pkl",
    "model_width": 640,
    "model_height": 360
}
```

## API Reference

### BallTracker

#### `__init__(config: dict = None)`

Initialize the ball tracker with optional configuration.

**Parameters:**

-   `config` (dict, optional): Configuration dictionary. If None, loads default config.

#### `process_frame(frame) -> tuple`

Process a single frame and return ball coordinates.

**Parameters:**

-   `frame` (numpy.ndarray): Input frame in BGR format (any resolution)

**Returns:**

-   `tuple`: (x, y) coordinates of the ball in original frame resolution, or (None, None) if not detected

#### `reset()`

Reset the tracker state by clearing the frame buffer.

## Package Structure

```
ball_detection_pipeline/
    __init__.py
    ball_tracker.py          # Main BallTracker class
    utils.py                 # Utility functions
    config.json              # Default configuration
    model/
        __init__.py
        model.py             # TrackNet model architecture
        weights/
            model.pkl        # Pre-trained model weights
```

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

This project is part of a thesis on Squash Coaching Copilot.
