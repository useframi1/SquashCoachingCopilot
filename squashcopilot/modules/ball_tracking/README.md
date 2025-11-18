# Ball Tracking Module

The ball tracking module is responsible for detecting the squash ball in video frames, tracking its trajectory over time, and detecting wall hits and racket hits.

## Overview

This module uses **TrackNet**, a deep learning model specifically designed for ball tracking in sports videos. It processes video frames to detect the ball position, smooths the trajectory to handle noise and occlusions, and identifies key events (wall hits and racket hits). The module is part of the larger SquashCoachingCopilot package and integrates with other modules through standardized data models.

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

## Components

### 1. BallTracker (`ball_tracker.py`)

The main class for ball detection and trajectory processing.

**Key Methods:**
- `set_is_black_ball(is_black: bool)`: Configure ball color for preprocessing
- `preprocess_frame(frame: Frame)`: Enhance frame using LAB color space and CLAHE
- `process_frame(input: BallTrackingInput)`: Detect ball position in a single frame
- `postprocess(input: BallPostprocessingInput)`: Smooth trajectory and remove outliers

**Processing Pipeline:**
1. **Preprocessing**: Convert to LAB color space and apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced ball visibility
2. **Inference**: Run TrackNet model to predict ball position and confidence
3. **Postprocessing**: Apply rolling window outlier detection to smooth trajectory

### 2. WallHitDetector (`wall_hit_detector.py`)

Detects when the ball hits the front wall using signal processing techniques.

**Algorithm:**
- Analyzes ball Y-coordinate trajectory (lower Y = higher on wall)
- Uses `scipy.signal.find_peaks` to detect local minima (wall contact points)
- Filters peaks using prominence and distance thresholds

**Parameters (from config):**
- `prominence`: 50 pixels - minimum prominence for peak detection
- `width`: 10 frames - minimum width of peak
- `min_distance`: 50 frames - minimum frames between consecutive hits

**Output:**
- Frame number of wall hit
- Position (pixel and meter coordinates)
- Prominence metric (strength of the hit)

### 3. RacketHitDetector (`racket_hit_detector.py`)

Detects racket hits by analyzing ball trajectory slope changes and player positions.

**Algorithm:**
1. Calculate trajectory slope over sliding window
2. Detect slope changes exceeding threshold (direction changes)
3. Attribute hits to players based on court position proximity
4. Filter false positives using temporal constraints

**Parameters (from config):**
- `slope_window`: 10 frames - window size for slope calculation
- `slope_threshold`: 3.0 - minimum slope change to detect hit
- `min_distance`: 15 frames - minimum frames between hits
- `lookback_frames`: 50 frames - how far to look back from wall hit

**Output:**
- Frame number of racket hit
- Player ID who made the hit
- Ball position at hit (pixel and meter coordinates)

### 4. TrackNetTracker (`model/tracknet_tracker.py`)

Core TrackNet model implementation for ball detection.

**Model Architecture:**
- Input: 640x360 RGB image
- Output: Heatmap with ball location probability
- Based on VGG-like architecture with encoder-decoder structure

## Data Models

The module uses standardized data models from `squashcopilot.common.models.ball`:

### Input Models
- **BallTrackingInput**: Single frame for ball detection
  - `frame`: Frame object with image data, frame number, and timestamp

- **BallPostprocessingInput**: Collection of raw detections for smoothing
  - `positions`: List of BallDetectionResult objects

### Output Models
- **BallDetectionResult**: Raw detection for a single frame
  - `position`: Point2D (x, y) or None if not detected
  - `confidence`: Detection confidence score (0-1)
  - `frame_number`: Frame index

- **BallTrajectory**: Smoothed trajectory with statistics
  - `positions`: List of smoothed Point2D positions
  - `gaps`: List of frame ranges with missing detections
  - `outliers_removed`: Count of outliers filtered out

- **WallHitDetectionResult**: Collection of wall hit events
  - `hits`: List of WallHit objects with frame, position (pixel & meter), prominence

- **RacketHitDetectionResult**: Collection of racket hit events
  - `hits`: List of RacketHit objects with frame, player_id, position (pixel & meter)

## Usage

### Using Standard Data Models

```python
from squashcopilot import BallTracker, WallHitDetector, RacketHitDetector
from squashcopilot import BallTrackingInput, BallPostprocessingInput
from squashcopilot import Frame

# Initialize tracker
tracker = BallTracker()
tracker.set_is_black_ball(is_black=True)

# Process frames
detections = []
for frame_data in video_frames:
    # Create input using standard Frame model
    input_data = BallTrackingInput(frame=frame_data)
    result = tracker.process_frame(input_data)
    detections.append(result)

# Postprocess trajectory
postprocess_input = BallPostprocessingInput(positions=detections)
trajectory = tracker.postprocess(postprocess_input)

# Detect wall hits (requires floor homography from court calibration)
wall_detector = WallHitDetector()
wall_hits = wall_detector.detect(
    ball_trajectory=trajectory,
    floor_homography=homography
)

# Detect racket hits (requires player trajectories)
racket_detector = RacketHitDetector()
racket_hits = racket_detector.detect(
    ball_trajectory=trajectory,
    player_trajectories=player_trajectories,
    wall_hits=wall_hits
)

print(f"Detected {len(wall_hits.hits)} wall hits")
print(f"Detected {len(racket_hits.hits)} racket hits")
```

### Integration with Other Modules

```python
from squashcopilot import Annotator

# The Annotator class orchestrates all modules including ball tracking
annotator = Annotator()
results = annotator.annotate_video("video.mp4", "output_directory")

# Access ball tracking results
ball_trajectory = results['ball_trajectory']
wall_hits = results['wall_hits']
racket_hits = results['racket_hits']
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

Configuration file: `squashcopilot/configs/ball_tracking.yaml`

```yaml
model:
  path: "path/to/tracknet_model.pth"
  input_height: 360
  input_width: 640
  device: "auto"  # auto, cuda, or cpu

postprocessing:
  outlier_detection:
    window: 20          # Rolling window size for outlier detection
    threshold: 80       # Distance threshold in pixels

wall_hit_detection:
  prominence: 50        # Minimum prominence for peak detection
  width: 10            # Minimum width of peak
  min_distance: 50     # Minimum frames between hits

racket_hit_detection:
  slope_window: 10      # Window for slope calculation
  slope_threshold: 3.0  # Minimum slope change
  min_distance: 15      # Minimum frames between hits
  lookback_frames: 50   # How far to look back from wall hit
```

### Configuration Parameters

**Model:**
- `path`: Path to TrackNet model weights (.pth file)
- `input_height/input_width`: TrackNet model input dimensions (640x360)
- `device`: Device for inference (`"auto"`, `"cuda"`, or `"cpu"`)

**Postprocessing:**
- `outlier_detection.window`: Rolling window size (20 frames)
- `outlier_detection.threshold`: Distance threshold in pixels (80px)

**Wall Hit Detection:**
- `prominence`: Minimum depth of valley in pixels (50px)
- `width`: Minimum width of valley in frames (10)
- `min_distance`: Minimum frames between consecutive hits (50)

**Racket Hit Detection:**
- `slope_window`: Number of frames to calculate slope over (10)
- `slope_threshold`: Minimum absolute slope in pixels/frame (3.0)
- `min_distance`: Minimum frames between consecutive hits (15)
- `lookback_frames`: How many frames to look back from wall hit (50)

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

## Module Structure

```
ball_tracking/
├── __init__.py                    # Package exports
├── ball_tracker.py                # Main BallTracker class
├── wall_hit_detector.py           # WallHitDetector class
├── racket_hit_detector.py         # RacketHitDetector class
├── utils.py                       # Utility functions
├── model/                         # Model implementations
│   ├── __init__.py
│   ├── tracknet_tracker.py        # TrackNetTracker class
│   └── model.py                   # BallTrackerNet architecture
└── tests/                         # Evaluation suite
    ├── evaluator.py               # Evaluation script
    ├── data/                      # Test videos
    └── outputs/                   # Results and metrics
```

## Testing and Evaluation

The `tests/` directory contains a comprehensive evaluation framework:

- **evaluator.py**: Measures tracking accuracy against ground truth
- **data/**: Test videos (video-1 through video-5)
- **outputs/**: Generated results and performance metrics

### Running Tests

```python
from squashcopilot.modules.ball_tracking.tests.evaluator import BallTrackingEvaluator

evaluator = BallTrackingEvaluator()
results = evaluator.evaluate(video_path="tests/data/video-1.mp4")
evaluator.generate_report(results)
```

## Algorithm Details

### Trajectory Smoothing
The postprocessing step uses a rolling window approach:
1. For each position, calculate distance to neighbors within window (20 frames)
2. If distance exceeds threshold (80px), mark as outlier
3. Replace outliers with interpolated values
4. Track gaps (consecutive missing detections)

### Wall Hit Detection
Wall hits are detected using peak detection on inverted Y-coordinates:
1. Invert Y-coordinates (lower Y = higher position on wall)
2. Apply Gaussian smoothing to reduce noise
3. Find peaks using `scipy.signal.find_peaks`
4. Filter by prominence (50px), width (10 frames), and minimum distance (50 frames)
5. Convert pixel positions to meter coordinates using floor homography

### Racket Hit Detection
Racket hits combine trajectory analysis with player position:
1. Calculate trajectory slope using finite differences over sliding window
2. Detect slope changes exceeding threshold (3.0 pixels/frame)
3. For each candidate hit, find nearest player
4. Verify player is within reasonable distance
5. Apply temporal filtering to prevent duplicates (min 15 frames apart)

## Performance Considerations

- **TrackNet Inference**: ~30-60 FPS on GPU, ~5-10 FPS on CPU
- **Postprocessing**: Real-time (>100 FPS)
- **Hit Detection**: Real-time (>200 FPS)

## Limitations

- May struggle with very fast ball movements (motion blur)
- Performance degrades with poor lighting conditions
- Black ball detection requires preprocessing configuration
- Occlusion handling relies on interpolation (may be inaccurate for long gaps)
- Wall hit detection assumes front wall is at top of frame (low Y values)

## Dependencies on Other Modules

This module integrates with:
- **Court Calibration**: Requires floor homography for coordinate transformation
- **Player Tracking**: Racket hit detection uses player positions for attribution
- Used by **Shot Classification**: Provides ball trajectory and hit events
- Used by **Rally Segmentation**: Provides ball Y positions for rally detection

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

MIT License

This module is part of the SquashCoachingCopilot thesis project at the American University in Cairo.
