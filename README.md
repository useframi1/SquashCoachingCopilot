# SquashCoachingCopilot

An automated computer vision system for comprehensive squash video analysis, providing batch processing of videos to extract player tracking, ball trajectory, shot classification, and tactical insights.

## Overview

SquashCoachingCopilot is a modular Python package that processes squash videos to extract detailed game analytics. It combines multiple deep learning models and computer vision techniques in a multi-stage batch processing pipeline to analyze player movements, ball trajectories, and shot types.

### Key Capabilities

-   **Court Calibration**: Automatic detection and mapping of court coordinates
-   **Player Tracking**: Multi-player detection with pose estimation (12 body keypoints)
-   **Ball Tracking**: High-accuracy ball detection using TrackNet
-   **Hit Detection**: Automatic detection of wall hits and racket hits
-   **Stroke Detection**: LSTM-based forehand/backhand classification using windowed keypoint sequences
-   **Shot Classification**: Rule-based shot type analysis (straight/cross-court, drop/drive)
-   **Rally Segmentation**: LSTM-based rally boundary detection
-   **Video Annotation**: Automated generation of annotated videos and CSV data exports

## Architecture

The package follows a **multi-stage batch processing pipeline** with standardized data models for inter-module communication:

```
Input Video
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 1: Court Calibration (first frame)       │
│     - Detect court elements (Roboflow API)      │
│     - Compute homographies (floor, wall)        │
│     - Detect wall color                         │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 2: Frame-by-Frame Tracking (parallel)    │
│     ┌─────────────────┐  ┌─────────────────┐    │
│     │  Player Track   │  │  Ball Track     │    │
│     │  (YOLO+ResNet)  │  │  (TrackNet)     │    │
│     │  - Detection    │  │  - Detection    │    │
│     │  - Pose         │  │  - Position     │    │
│     │  - Re-ID        │  │  - Confidence   │    │
│     └─────────────────┘  └─────────────────┘    │
│           ↓                      ↓              │
│     Raw Player Data        Raw Ball Data        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 3: Trajectory Postprocessing (parallel)  │
│     ┌─────────────────┐  ┌─────────────────┐    │
│     │  Player Post.   │  │  Ball Post.     │    │
│     │  - Smoothing    │  │  - Smoothing    │    │
│     │  - Interpolate  │  │  - Gap filling  │    │
│     │  - Transform    │  │  - Outlier rem. │    │
│     └─────────────────┘  └─────────────────┘    │
│           ↓                      ↓              │
│   Player Trajectories      Ball Trajectory      │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 4: Rally Segmentation                    │
│     - LSTM-based rally detection                │
│     - Input: Ball Y + Player positions          │
│     - Output: Rally segments (start, end)       │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 5: Per-Rally Hit Detection               │
│     For each rally segment:                     │
│     ┌───────────────────────────────────────┐   │
│     │  Wall Hit Detection                   │   │
│     │  - Signal processing on ball Y        │   │
│     │  - Peak detection                     │   │
│     └───────────────────────────────────────┘   │
│     ┌───────────────────────────────────────┐   │
│     │  Racket Hit Detection                 │   │
│     │  - Trajectory slope analysis          │   │
│     │  - Player attribution                 │   │
│     └───────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 6: Shot & Stroke Classification          │
│     ┌───────────────────────────────────────┐   │
│     │  Stroke Detection (LSTM)              │   │
│     │  - Forehand/backhand classification   │   │
│     │  - 31-frame windowed prediction       │   │
│     │  - Hip-torso normalized keypoints     │   │
│     └───────────────────────────────────────┘   │
│     ┌───────────────────────────────────────┐   │
│     │  Shot Classification (Rule-based)     │   │
│     │  - Direction (straight/cross-court)   │   │
│     │  - Depth (drop/drive)                 │   │
│     └───────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 7: Analysis & Output Generation          │
│     - Statistical analysis                      │
│     - Annotated video (MP4)                     │
│     - Frame-by-frame data (CSV)                 │
│     - Shot/rally statistics (JSON)              │
└─────────────────────────────────────────────────┘
```

**Processing Model**: Batch processing - the entire video is processed through each stage sequentially. Stages 2 and 3 leverage parallelism for player and ball processing, but the overall pipeline is not real-time.

## Project Structure

```
SquashCoachingCopilot/
├── squashcopilot/                    # Main package
│   ├── __init__.py                   # Package exports
│   ├── common/                       # Shared utilities and data models
│   │   ├── types/                    # Base types (Frame, Point2D, BoundingBox, etc.)
│   │   ├── models/                   # Data models for all modules
│   │   │   ├── ball.py               # Ball tracking models
│   │   │   ├── court.py              # Court calibration models
│   │   │   ├── player.py             # Player tracking models
│   │   │   ├── rally.py              # Rally segmentation models
│   │   │   ├── stroke.py             # Stroke detection models
│   │   │   └── shot.py               # Shot classification models
│   │   ├── constants.py              # Global constants (COCO keypoints, etc.)
│   │   └── utils.py                  # Utilities (config loading, path management)
│   │
│   ├── configs/                      # YAML configuration files
│   │   ├── annotation.yaml           # Annotation pipeline config
│   │   ├── ball_tracking.yaml        # Ball tracking config
│   │   ├── court_calibration.yaml    # Court calibration config
│   │   ├── player_tracking.yaml      # Player tracking config
│   │   ├── rally_state_detection.yaml# Rally segmentation config
│   │   ├── shot_type_classification.yaml  # Shot classification config
│   │   └── stroke_detection.yaml     # Stroke detection config
│   │
│   ├── annotation/                   # Video annotation pipeline
│   │   ├── annotator.py              # Main Annotator class (pipeline orchestrator)
│   │   ├── data/                     # Input videos
│   │   └── annotations/              # Output annotations
│   │
│   └── modules/                      # Processing modules
│       ├── ball_tracking/            # Ball detection and hit detection
│       │   ├── ball_tracker.py       # Main tracker
│       │   ├── wall_hit_detector.py  # Wall hit detection
│       │   ├── racket_hit_detector.py# Racket hit detection
│       │   ├── model/                # TrackNet implementation
│       │   ├── tests/                # Evaluation suite
│       │   └── README.md
│       │
│       ├── court_calibration/        # Court detection and calibration
│       │   ├── court_calibrator.py   # Main calibrator
│       │   ├── tests/                # Evaluation suite
│       │   └── README.md
│       │
│       ├── player_tracking/          # Player detection and tracking
│       │   ├── player_tracker.py     # Main tracker
│       │   ├── model/                # YOLO + ResNet50
│       │   ├── tests/                # Evaluation suite
│       │   └── README.md
│       │
│       ├── rally_state_detection/    # Rally segmentation
│       │   ├── rally_state_detector.py  # Main detector
│       │   ├── models/               # LSTM implementation
│       │   ├── train_model.py        # Training utilities
│       │   ├── tests/                # Evaluation suite
│       │   └── README.md
│       │
│       ├── shot_type_classification/ # Shot analysis
│       │   ├── shot_classifier.py    # Main classifier
│       │   ├── tests/                # Evaluation suite
│       │   └── README.md
│       │
│       └── stroke_detection/         # Stroke type detection
│           ├── stroke_detector.py    # Main StrokeDetector class
│           ├── model/                # LSTM implementation
│           │   ├── lstm_classifier.py  # LSTMStrokeClassifier
│           │   └── weights/          # Model checkpoints
│           ├── train_model.py        # Training script with StrokeTrainer
│           ├── tests/                # Evaluation suite
│           │   ├── evaluator.py      # StrokeDetectionEvaluator
│           │   ├── data/             # Training/test annotations
│           │   └── outputs/          # Evaluation results
│           └── README.md
│
├── README.md                         # This file
└── pyproject.toml                    # Package configuration
```

## Installation

### Prerequisites

-   Python >= 3.8
-   CUDA-compatible GPU (optional, but highly recommended for faster batch processing)

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/SquashCoachingCopilot.git
cd SquashCoachingCopilot

# Install in development mode
pip install -e .
```

### Dependencies

Core dependencies:

-   **PyTorch**: Deep learning models (TrackNet, LSTM)
-   **Ultralytics**: YOLOv8 for player detection
-   **OpenCV**: Image processing and video I/O
-   **NumPy**: Numerical operations
-   **SciPy**: Signal processing (hit detection, interpolation)
-   **inference**: Roboflow API client (court calibration)

## Quick Start

### Complete Video Annotation

```python
from squashcopilot import Annotator

# Initialize annotator
annotator = Annotator()

# Process video
results = annotator.annotate_video(
    video_path="match.mp4",
    output_dir="output"
)

# Outputs:
# - output/match_annotated.mp4  (annotated video)
# - output/match_annotations.csv (frame-by-frame data)
```

### Module-by-Module Usage

```python
from squashcopilot import (
    CourtCalibrator, PlayerTracker, BallTracker,
    WallHitDetector, RacketHitDetector, ShotClassifier,
    RallyStateDetector, StrokeDetector
)
from squashcopilot import Frame
import cv2

# Initialize modules
court_calibrator = CourtCalibrator()
player_tracker = PlayerTracker()
ball_tracker = BallTracker()

# Read video
cap = cv2.VideoCapture("match.mp4")

# Calibrate court (first frame)
ret, frame_img = cap.read()
frame = Frame(image=frame_img, frame_number=0, timestamp=0.0)
calibration = court_calibrator.process_frame(frame)

# Set homographies
player_tracker.floor_homography = calibration.floor_homography
ball_tracker.is_black_ball = calibration.wall_color.is_dark_wall

# Process frames
while cap.isOpened():
    ret, frame_img = cap.read()
    if not ret:
        break

    frame = Frame(image=frame_img, frame_number=..., timestamp=...)

    # Track players and ball
    player_result = player_tracker.process_frame(frame)
    ball_result = ball_tracker.process_frame(frame)

    # ... (continue processing)

cap.release()
```

## Data Models

All modules use standardized data models from `squashcopilot.common.models`:

### Core Types (`common/types/`)

-   **Frame**: Video frame with image, frame number, timestamp
-   **Point2D**: 2D point with x, y coordinates
-   **BoundingBox**: Rectangular region (x1, y1, x2, y2)
-   **Homography**: 3x3 transformation matrix
-   **Keypoints**: Collection of named 2D points with confidence scores

### Module-Specific Models (`common/models/`)

-   **ball.py**: Ball detection, trajectories, wall hits, racket hits
-   **court.py**: Court calibration, homographies, wall color
-   **player.py**: Player detection, tracking, keypoints, trajectories
-   **rally.py**: Rally segments and statistics
-   **stroke.py**: Stroke detection and sequences
-   **shot.py**: Shot classification and statistics

## Data Flow

The pipeline processes videos in distinct stages:

1. **Input**: Video file → Frame objects
2. **Court Calibration** (Stage 1): First frame → CourtCalibrationResult (homographies)
3. **Tracking** (Stage 2): All frames → Raw PlayerDetectionResult + BallDetectionResult (parallel)
4. **Postprocessing** (Stage 3): Raw detections → PlayerTrajectory + BallTrajectory (parallel)
5. **Rally Segmentation** (Stage 4): Trajectories → RallySegmentationResult
6. **Hit Detection** (Stage 5): Per-rally trajectory analysis → WallHitDetectionResult + RacketHitDetectionResult
7. **Classification** (Stage 6): Hits + Trajectories → StrokeDetectionResult + ShotClassificationResult
8. **Analysis & Output** (Stage 7): Annotated video + CSV + rally/shot statistics

**Note**: Each stage completes fully before the next stage begins (batch processing).

## Configuration

All modules are configured via YAML files in `squashcopilot/configs/`. See individual module READMEs for detailed configuration options.

## Module Documentation

Each module has detailed documentation:

-   [Ball Tracking](squashcopilot/modules/ball_tracking/README.md)
-   [Court Calibration](squashcopilot/modules/court_calibration/README.md)
-   [Player Tracking](squashcopilot/modules/player_tracking/README.md)
-   [Rally State Detection](squashcopilot/modules/rally_state_detection/README.md)
-   [Shot Type Classification](squashcopilot/modules/shot_type_classification/README.md)
-   [Stroke Detection](squashcopilot/modules/stroke_detection/README.md)

## CSV Output Format

The annotation pipeline generates frame-by-frame CSV files with the following columns:

-   **Frame metadata**: frame, timestamp
-   **Player data** (per player):
    -   Position (pixel): player{N}\_x, player{N}\_y
    -   Position (meters): player{N}\_x_m, player{N}\_y_m
    -   Bounding box: player{N}_bbox_{x1,y1,x2,y2}
    -   Keypoints (12 body points): player{N}_{keypoint}\_x, player{N}_{keypoint}\_y
    -   Confidence: player{N}\_confidence
-   **Ball data**:
    -   Position: ball_x, ball_y
    -   Position (meters): ball_x_m, ball_y_m
    -   Confidence: ball_confidence
-   **Hit events**:
    -   is_wall_hit, wall_hit_x, wall_hit_y, wall_hit_x_m, wall_hit_y_m
    -   is_racket_hit, racket_hit_player_id, racket_hit_x, racket_hit_y

## Performance

Typical performance on a system with NVIDIA RTX 3090:

| Module                    | GPU FPS  | CPU FPS | Notes                          |
| ------------------------- | -------- | ------- | ------------------------------ |
| Court Calibration         | N/A      | N/A     | API call ~1s, once per video   |
| Player Tracking           | 20-40    | 3-5     | Stage 2                        |
| Ball Tracking             | 30-60    | 5-10    | Stage 2 (parallel with player) |
| Player Postprocessing     | >100     | >100    | Stage 3                        |
| Ball Postprocessing       | >100     | >100    | Stage 3 (parallel with player) |
| Rally Segmentation        | 500-1000 | 100-200 | Stage 4 (batch LSTM)           |
| Wall/Racket Hit Detection | >200     | >200    | Stage 5 (per rally)            |
| Stroke Detection          | 500-1000 | 100-200 | Stage 6 (windowed LSTM)        |
| Shot Classification       | >1000    | >1000   | Stage 6 (rule-based)           |

**Complete Pipeline Throughput**:

-   **GPU**: ~15-25 FPS equivalent (dominated by tracking stages)
-   **CPU**: ~2-4 FPS equivalent (dominated by tracking stages)
-   **Processing Time**: A 5-minute video takes approximately 1-2 minutes on GPU, 5-10 minutes on CPU

**Note**: Pipeline is batch-oriented, not real-time. Stages 2 and 3 can process frames in parallel for player/ball, but overall flow is sequential by stage.

## Limitations

-   **Batch Processing Only**: Not designed for real-time analysis during live matches
-   Requires clear view of court (static camera recommended)
-   Performance degrades with poor lighting or motion blur
-   Court calibration requires internet connection (Roboflow API)
-   Player tracking assumes two players maximum
-   Shot classification uses rule-based approach (may not capture all shot variations)
-   Models trained on specific datasets may require fine-tuning for different courts/cameras
-   Rally segmentation must complete before hit detection (cannot be done in parallel)

## Future Improvements

-   Optimized batch processing with GPU multi-streaming
-   Additional shot types (lobs, boasts, kills, volley)
-   Player performance metrics (movement patterns, court coverage, positioning)
-   Shot quality assessment (pace, accuracy, placement)
-   Multi-camera support for court coverage
-   Real-time streaming variant for live match analysis
-   Incremental processing for partial video analysis

## Contributing

This is a thesis project. For questions or collaboration inquiries, please contact the author.

## Citation

If you use this work in your research, please cite:

```
@mastersthesis{elhagg2025squashcopilot,
  author = {Elhagg, Youssef},
  title = {SquashCoachingCopilot: Automated Computer Vision System for Squash Video Analysis},
  school = {American University in Cairo},
  year = {2025}
}
```

## License

MIT License

## Author

**Youssef Elhagg**
Email: yousseframi@aucegypt.edu
American University in Cairo

Thesis Project: Squash Coaching Copilot
Department: Computer Science and Engineering

## Acknowledgments

This project builds upon several open-source projects:

-   TrackNet (ball tracking)
-   YOLOv8 (player detection)
-   Roboflow (court detection)

Special thanks to the thesis advisors and the squash community for providing test data and feedback.
