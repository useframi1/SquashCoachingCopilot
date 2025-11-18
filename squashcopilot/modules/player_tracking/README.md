# Player Tracking Module

The player tracking module detects and tracks two squash players in video footage, extracting their positions, bounding boxes, and pose keypoints over time.

## Overview

This module provides comprehensive player tracking for squash videos by:

-   **Player Detection**: Using YOLOv8 for robust person detection
-   **Pose Estimation**: Extracting 17 COCO keypoints for body pose
-   **Re-identification**: Using ResNet50 features to maintain player identities
-   **Trajectory Smoothing**: Cubic spline interpolation for smooth position tracking
-   **Court Masking**: Filtering out audience/non-court detections
-   **Coordinate Transformation**: Converting pixel positions to real-world meters

The module is part of the SquashCoachingCopilot package and provides essential player data used by stroke detection, shot classification, and rally analysis modules.

## Features

-   **Real-time player detection** using YOLOv8
-   **COCO pose keypoints** (17 points per player)
-   **Player re-identification** with ResNet50 feature extraction
-   **Robust tracking** combining appearance and position-based matching
-   **Trajectory postprocessing** with cubic spline interpolation
-   **Keypoint interpolation** for missing detections
-   **Court masking** to filter non-player detections
-   **Coordinate transformation** via floor homography
-   **GPU acceleration** support

## Components

### PlayerTracker (`player_tracker.py`)

The main class for player detection, tracking, and postprocessing.

**Key Methods:**
- `process_frame(input: PlayerTrackingInput)`: Detect players in a single frame
- `postprocess(input: PlayerPostprocessingInput)`: Smooth trajectories and interpolate keypoints
- `reset()`: Reset tracker state for new video
- `floor_homography`: Property to access/set the floor homography matrix
- `wall_homography`: Property to access/set the wall homography matrix

**Processing Pipeline:**
1. **Detection**: YOLO detects all persons in frame
2. **Court Filtering**: Apply court mask to remove audience
3. **Pose Estimation**: Extract 17 COCO keypoints per detection
4. **Feature Extraction**: ResNet50 generates appearance features
5. **Player Assignment**: Match detections to player IDs (1, 2)
6. **Position Calculation**: Compute bottom-center foot position
7. **Coordinate Transform**: Convert to real-world meters using homography

**Postprocessing:**
- Smooth position trajectories using cubic spline interpolation
- Fill gaps in keypoint data through temporal interpolation
- Handle missing detections gracefully

### Dependencies

-   **Ultralytics**: YOLOv8 for detection and pose estimation
-   **PyTorch**: ResNet50 for re-identification
-   **NumPy**: Numerical operations
-   **SciPy**: Cubic spline interpolation
-   **OpenCV**: Image processing and coordinate transformation

## Data Models

The module uses standardized data models from `squashcopilot.common.models.player`:

### Input Models
- **PlayerTrackingInput**: Single frame for player detection
  - `frame`: Frame object with image data
  - `court_mask`: Optional binary mask to filter non-court regions

- **PlayerPostprocessingInput**: Collection of raw detections for smoothing
  - `detections`: Dictionary mapping frames to PlayerDetectionResult
  - `floor_homography`: Homography for coordinate transformation

### Output Models
- **PlayerDetectionResult**: Raw detection for a single frame
  - `detections`: Dictionary mapping player IDs (1, 2) to detection data
  - Each detection contains:
    - `position`: Point2D (x, y) in pixels
    - `position_meters`: Point2D in real-world meters
    - `bbox`: BoundingBox (x1, y1, x2, y2)
    - `keypoints`: PlayerKeypointsData (17 COCO points)
    - `confidence`: Detection confidence score (0-1)

- **PlayerKeypointsData**: COCO 17-point pose keypoints
  - `keypoints`: List of Point2D for each body part
  - `confidences`: Confidence score per keypoint
  - Keypoint order: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

- **PlayerTrajectory**: Smoothed trajectory for one player
  - `positions`: List of smoothed Point2D positions (pixels)
  - `positions_meters`: List of positions in meters
  - `keypoints`: List of interpolated PlayerKeypointsData
  - `bboxes`: List of BoundingBox objects
  - `confidences`: List of confidence scores

- **PlayerPostprocessingResult**: Dictionary of trajectories
  - Maps player IDs (1, 2) to PlayerTrajectory objects

## Usage

### Using Standard Data Models

```python
from squashcopilot import PlayerTracker, CourtCalibrator
from squashcopilot import PlayerTrackingInput, PlayerPostprocessingInput, Frame

# Initialize tracker and calibrator
tracker = PlayerTracker()
calibrator = CourtCalibrator()

# Calibrate court on first frame
first_frame = Frame(image=first_frame_img, frame_number=0, timestamp=0.0)
calibration = calibrator.process_frame(CourtCalibrationInput(frame=first_frame))

# Set homography for coordinate transformation
tracker.floor_homography = calibration.floor_homography

# Process frames
detections = {}
for frame_num, frame_img in enumerate(video_frames):
    frame = Frame(image=frame_img, frame_number=frame_num, timestamp=frame_num/fps)
    input_data = PlayerTrackingInput(frame=frame)
    result = tracker.process_frame(input_data)
    detections[frame_num] = result

# Postprocess trajectories
postprocess_input = PlayerPostprocessingInput(
    detections=detections,
    floor_homography=calibration.floor_homography
)
trajectories = tracker.postprocess(postprocess_input)

# Access smoothed data
player1_trajectory = trajectories[1]
player2_trajectory = trajectories[2]

print(f"Player 1 positions: {len(player1_trajectory.positions)} frames")
print(f"Player 2 positions: {len(player2_trajectory.positions)} frames")
```

### Integration with Other Modules

```python
from squashcopilot import Annotator

# Annotator handles player tracking automatically
annotator = Annotator()
results = annotator.annotate_video("video.mp4", "output_directory")

# Access player trajectories
player_trajectories = results['player_trajectories']
```

## API Reference

### PlayerTracker

The main class for player detection and tracking.

#### `__init__(config: dict = None)`

Initialize the player tracker.

**Parameters:**

-   `config` (dict, optional): Configuration dictionary. If None, loads from package's `config.json`.

**Example:**

```python
# Use default config
tracker = PlayerTracker()

# Use custom config
custom_config = {
    "models": {
        "yolo_model": "model/weights/yolov8m.pt",
        "reid_feature_size": 512,
        "reid_input_size": [224, 224]
    },
    "tracker": {
        "max_history": 30,
        "reid_threshold": 0.07,
        "reid_weight": 0.1,
        "position_weight": 0.9
    }
}
tracker = PlayerTracker(config=custom_config)
```

#### `process_frame(frame)`

Process a single frame and return player tracking information.

**Parameters:**

-   `frame` (numpy.ndarray): BGR image from OpenCV

**Returns:**

-   `dict`: Dictionary with tracking results for both players:
    ```python
    {
        1: {
            "position": (x, y),      # Bottom-center position or None
            "bbox": [x1, y1, x2, y2], # Bounding box or None
            "confidence": float       # Detection confidence or None
        },
        2: {
            "position": (x, y),      # Bottom-center position or None
            "bbox": [x1, y1, x2, y2], # Bounding box or None
            "confidence": float       # Detection confidence or None
        }
    }
    ```

#### `reset()`

Reset tracker state. Useful when processing a new video.

**Example:**

```python
tracker.reset()
```

## Configuration

Configuration file: `squashcopilot/configs/player_tracking.yaml`

```yaml
model:
  yolo_model: "yolov8m-pose.pt"
  device: "auto"  # auto, cuda, or cpu
  confidence_threshold: 0.5

reid:
  feature_size: 512
  input_size: [224, 224]
  threshold: 0.07
  weight: 0.1  # Appearance matching weight

tracker:
  max_history: 30
  position_weight: 0.9  # Position-based matching weight

preprocessing:
  use_court_mask: true
  mask_dilation_kernel: 15

postprocessing:
  smoothing_window: 5
  interpolation_method: "cubic"  # linear, cubic
  max_gap_frames: 30  # Maximum gap to interpolate
```

### Configuration Parameters

**Model:**
- `yolo_model`: YOLOv8 model variant (yolov8n/s/m/l/x-pose.pt)
- `device`: Inference device (auto/cuda/cpu)
- `confidence_threshold`: Minimum confidence for person detection (0.5)

**Re-identification:**
- `feature_size`: ResNet50 feature dimension (512)
- `input_size`: Input size for feature extractor [224, 224]
- `threshold`: Cosine distance threshold for matching (0.07)
- `weight`: Weight for appearance-based matching (0.1)

**Tracker:**
- `max_history`: Maximum frames of tracking history (30)
- `position_weight`: Weight for position-based matching (0.9)

**Preprocessing:**
- `use_court_mask`: Enable court masking to filter audience (true)
- `mask_dilation_kernel`: Kernel size for mask dilation (15)

**Postprocessing:**
- `smoothing_window`: Window size for trajectory smoothing (5)
- `interpolation_method`: Spline type (cubic/linear)
- `max_gap_frames`: Maximum frames to interpolate missing detections (30)

## Algorithm Details

### Player Detection and Tracking
1. **YOLO Detection**: YOLOv8-pose detects persons and extracts 17 COCO keypoints
2. **Court Masking**: Apply court mask (from court calibration) to filter audience
3. **Feature Extraction**: ResNet50 extracts 512-d appearance features from crops
4. **Player Assignment**:
   - **First Frame**: Assign IDs based on left-right position
   - **Subsequent Frames**: Hybrid matching:
     - Compute cosine distance for appearance similarity
     - Compute Euclidean distance for position proximity
     - Combined score = `reid_weight * appearance + position_weight * position`
     - Assign to player with lowest combined distance
5. **Feature Update**: Exponential moving average of appearance features

### Trajectory Postprocessing
1. **Position Smoothing**:
   - Extract raw positions for each player
   - Apply cubic spline interpolation to smooth trajectory
   - Handle missing frames through interpolation (up to max_gap_frames)

2. **Keypoint Interpolation**:
   - For frames with missing keypoints, interpolate from neighboring frames
   - Use temporal coherence to maintain pose consistency
   - Apply confidence-weighted interpolation

3. **Coordinate Transformation**:
   - Transform pixel positions to meters using floor homography
   - Store both pixel and meter coordinates

### COCO Keypoint Order
The 17 COCO keypoints are:
1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle

## Module Structure

```
player_tracking/
├── __init__.py                # Package exports
├── player_tracker.py          # Main PlayerTracker class
├── model/                     # Model implementations
│   ├── __init__.py
│   ├── yolo_detector.py       # YOLO wrapper
│   └── reid_extractor.py      # ResNet50 feature extractor
└── tests/                     # Evaluation suite
    ├── evaluator.py           # Evaluation script
    ├── data/                  # Test videos
    └── outputs/               # Results and metrics
```

## Testing and Evaluation

The `tests/` directory contains evaluation tools:

- **evaluator.py**: Tests tracking accuracy and identity consistency
- **data/**: Test videos with ground truth player positions
- **outputs/**: Tracking visualizations and metrics

### Running Tests

```python
from squashcopilot.modules.player_tracking.tests.evaluator import PlayerTrackingEvaluator

evaluator = PlayerTrackingEvaluator()
results = evaluator.evaluate(video_path="tests/data/video-1.mp4")
evaluator.generate_report(results)
```

## Performance Considerations

- **YOLO Inference**: ~20-40 FPS on GPU (YOLOv8m-pose), ~3-5 FPS on CPU
- **ResNet50 Re-ID**: ~50 FPS on GPU, ~10 FPS on CPU
- **Postprocessing**: Real-time (>100 FPS)
- **Recommendation**: Use GPU for real-time processing

## Limitations

- Requires two players visible in frame for accurate tracking
- Identity switches may occur during occlusions or close proximity
- Pose estimation quality degrades with motion blur
- Court mask required for accurate filtering of audience
- Interpolation quality depends on gap length and motion complexity

## Dependencies on Other Modules

This module integrates with:
- **Court Calibration**: Requires floor homography for coordinate transformation
- Used by **Racket Hit Detection**: Provides player positions for hit attribution
- Used by **Stroke Detection**: Provides keypoints for stroke classification
- Used by **Shot Classification**: Provides player positions for shot analysis
- Used by **Rally Segmentation**: Provides player positions for rally detection

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

MIT License

This module is part of the SquashCoachingCopilot thesis project at the American University in Cairo.
