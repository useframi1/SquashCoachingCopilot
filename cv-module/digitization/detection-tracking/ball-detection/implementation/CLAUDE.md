# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ball tracking system for squash videos using a deep learning model (TrackNet). The system processes video frames to detect and track the ball position in real-time, generating tracking visualizations and performance metrics.

## Architecture

### Core Components

1. **BallTrackerNet** ([model.py](model.py)) - PyTorch CNN model for ball detection
   - Encoder-decoder architecture with VGG-like encoder and upsampling decoder
   - Input: 9 channels (3 consecutive BGR frames concatenated)
   - Output: 256-channel heatmap (360x640) representing ball probability
   - The model requires exactly 3 consecutive frames for inference

2. **BallTracker** ([ball_tracker.py](ball_tracker.py)) - Real-time tracking interface
   - Maintains a sliding window buffer of the last 3 frames (using `deque`)
   - Handles frame preprocessing (resizing to 640x360) and coordinate scaling
   - Returns ball positions in original frame coordinates
   - Automatically selects CUDA device if available

3. **BallTrackingEvaluator** ([evaluator.py](evaluator.py)) - Video processing and evaluation pipeline
   - Processes entire videos and generates annotated output
   - Calculates detection rate and velocity metrics
   - Saves tracking visualizations, plots, and CSV metrics
   - Entry point for evaluation workflow

4. **Utilities** ([general.py](general.py))
   - `postprocess()`: Converts model heatmap to (x,y) coordinates using Hough circle detection
   - `train()` and `validate()`: Training/validation loops (for model development)
   - `load_config()`: JSON configuration loader

### Data Flow

```
Video Frame → BallTracker.process_frame() → Model Inference → postprocess() → (x,y) coordinates
                     ↓
              Frame Buffer (3 frames)
                     ↓
              Concatenate & Normalize
                     ↓
              BallTrackerNet
                     ↓
              Heatmap (360x640)
                     ↓
              Coordinate Extraction
```

### Configuration System

All runtime parameters are controlled via [config.json](config.json):

- **model**: Model path, device selection, input dimensions
- **video**: Input video path, max frames to process
- **tracking**: Trace visualization parameters (length, color, thickness)
- **output**: Save options for video, plots, and metrics

## Running the Code

### Evaluate ball tracking on a video

```bash
python3 evaluator.py
```

This will:
1. Load the model from `model_best.pt`
2. Process the video specified in `config.json`
3. Generate annotated video in `tracking_results/`
4. Save position/velocity plots and metrics CSV

### Test model architecture

```bash
python3 model.py
```

Runs a forward pass test with random input to verify model shape.

## Key Implementation Details

### Coordinate Scaling

The model outputs coordinates at 2x the input resolution (due to `postprocess()` scale factor):
- Model input: 640x360
- Heatmap output: 640x360
- Coordinates after postprocess: scaled by 2x
- `BallTracker._get_scaled_coordinates()` handles conversion back to original video resolution

### Frame Buffer Management

The tracker requires 3 consecutive frames. The buffer ordering in inference is:
- `frames_list[0]`: pre-previous frame (oldest)
- `frames_list[1]`: previous frame
- `frames_list[2]`: current frame (newest)

These are concatenated as (current, previous, pre-previous) for model input.

### Device Handling

Setting `device: "auto"` in config automatically selects CUDA if available, otherwise falls back to CPU.

## Dependencies

- PyTorch (with CUDA support recommended)
- OpenCV (cv2)
- NumPy
- SciPy (for distance calculations)
- Matplotlib (for plotting)

## Project Context

This module is part of a larger squash coaching system located at:
`/thesis/SquashCoachingCopilot/cv-module/digitization/detection-tracking/ball-detection/`

The implementation sits alongside model weights (`model/`) and research papers (`papers/`).
