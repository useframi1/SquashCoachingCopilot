# Squash Coaching Pipeline

A comprehensive video analysis pipeline for squash coaching that processes match videos and generates detailed performance insights.

## Overview

This pipeline analyzes squash match videos to provide:
- **Player tracking** with pose estimation
- **Ball detection** and trajectory tracking
- **Court calibration** using computer vision
- **Rally segmentation** and state detection
- **Movement analysis** (speed, distance, court coverage)
- **Coaching insights** and performance comparisons

Built with a clean three-layer architecture following SOLID principles.

## Architecture

### Layer 1: Orchestration (`orchestration/`)
- **PipelineOrchestrator**: Coordinates sub-pipelines (court detection, player tracking, ball detection, rally state)
- **Visualizer**: Renders bounding boxes, keypoints, skeleton, ball position, and rally state

### Layer 2: Data Collection (`data/`)
- **DataCollector**: Aggregates raw pipeline outputs, validates data quality, applies post-processing
- **DataModels**: Structured data classes (FrameData, PlayerData, BallData, CourtData)
- **Validators**: Data quality validation (confidence thresholds, temporal consistency)
- **PostProcessors**: Temporal smoothing and missing data interpolation

### Layer 3: Analysis (`analysis/`)
- **CoachingAnalyzer**: High-level match analysis, coaching insights, and recommendations
- **MovementAnalyzer**: Speed, acceleration, distance, court coverage, sprint detection
- **RallyAnalyzer**: Rally segmentation, duration, intensity, shot counting
- **Metrics**: Utility functions for metric computation

## Quick Start

### Option 1: Simple Usage (Using main.py)

```bash
# Configure video path in config.py, then run:
python main.py
```

### Option 2: Using Pipeline API

```python
from pipeline import Pipeline
from config import PipelineConfig

# Create configuration
config = PipelineConfig(
    video_path="your_video.mp4",
    output_path="output_video.mp4",
    analysis_output_path="match_analysis",
    display=True
)

# Initialize and run pipeline
pipeline = Pipeline(config=config)
pipeline.run()
```

### Option 3: Advanced Usage (Direct Architecture Access)

```python
from orchestration import PipelineOrchestrator, Visualizer
from data import DataCollector
from analysis import CoachingAnalyzer
from video_io import VideoReader, VideoWriter

# 1. Configure components
data_collector = DataCollector(
    enable_smoothing=True,
    smoothing_window=5,
    enable_validation=True,
    min_confidence=0.3,
    max_position_change=200.0,
    handle_missing_data=True
)

visualizer = Visualizer(
    show_court_keypoints=True,
    show_player_keypoints=True,
    show_player_bbox=True,
    show_ball=True,
    show_rally_state=True,
    keypoint_confidence_threshold=0.5
)

orchestrator = PipelineOrchestrator(
    data_collector=data_collector,
    visualizer=visualizer
)

# 2. Process video
with VideoReader("video.mp4") as video_reader:
    metadata = video_reader.get_metadata()

    with VideoWriter("output.mp4", metadata) as video_writer:
        def write_frame(frame_data, annotated_frame):
            video_writer.write(annotated_frame)

        orchestrator.process_frames(
            frames_iterator=video_reader.frames(),
            video_metadata={
                "fps": metadata.fps,
                "width": metadata.width,
                "height": metadata.height,
                "total_frames": metadata.total_frames
            },
            display=True,
            on_frame_processed=write_frame
        )

# 3. Analyze data
frames = orchestrator.get_collected_data()
analyzer = CoachingAnalyzer(fps=metadata.fps)

# Get comprehensive match analysis
match_analysis = analyzer.analyze_match(frames)

# Compare players
comparison = analyzer.compare_players(frames)

# Export results
analyzer.export_analysis(frames, "analysis.json", format="json")
analyzer.export_analysis(frames, "summary.txt", format="summary")
```

## Features

### Data Collection & Processing
- ✅ Temporal smoothing of positions
- ✅ Data validation and quality checks
- ✅ Missing data interpolation
- ✅ Structured data models

### Movement Analysis
- ✅ Speed and acceleration profiles
- ✅ Total distance traveled
- ✅ Court coverage percentage
- ✅ Sprint detection
- ✅ Direction changes
- ✅ Positioning analysis

### Rally Analysis
- ✅ Automatic rally segmentation
- ✅ Rally duration and intensity
- ✅ Shot count estimation
- ✅ Rally statistics

### Coaching Insights
- ✅ Performance comparison
- ✅ Automated recommendations
- ✅ JSON and text export
- ✅ Player-specific analysis

## Configuration

The pipeline uses a centralized configuration system in [config.py](config.py):

### PipelineConfig
```python
from config import PipelineConfig, DataCollectorConfig, VisualizerConfig

config = PipelineConfig(
    video_path="video.mp4",              # Input video path
    output_path="output.mp4",            # Output video path (None to skip)
    analysis_output_path="analysis",     # Analysis export path prefix
    display=True,                        # Show real-time video display

    # Component configurations
    data_collector=DataCollectorConfig(
        enable_smoothing=True,           # Apply temporal smoothing
        smoothing_window=5,              # Moving average window size
        enable_validation=True,          # Validate data quality
        min_confidence=0.3,              # Minimum detection confidence
        max_position_change=200.0,       # Max pixels/frame for validation
        handle_missing_data=True,        # Interpolate missing detections
        max_interpolation_frames=10      # Max gap to interpolate
    ),

    visualizer=VisualizerConfig(
        show_court_keypoints=True,       # Display court calibration
        show_player_keypoints=True,      # Display player pose
        show_player_bbox=True,           # Display bounding boxes
        show_ball=True,                  # Display ball position
        show_rally_state=True,           # Display rally state overlay
        keypoint_confidence_threshold=0.5 # Min confidence for keypoint display
    )
)
```

## Design Principles

### Separation of Concerns
Each layer has a single, well-defined responsibility:
- **Orchestration**: Manages flow, NOT data processing
- **Data Collection**: Manages data quality, NOT insights
- **Analysis**: Computes insights, NOT data collection

### Dependency Inversion
Layers communicate through data models, not concrete implementations.

### Open/Closed Principle
Easy to extend with new analyzers without modifying existing code.

### Testability
Each layer can be unit tested independently.

## File Structure

```
full-pipeline/
├── orchestration/
│   ├── __init__.py
│   ├── pipeline_orchestrator.py    # Coordinates sub-pipelines
│   └── visualizer.py                # Rendering and visualization
├── data/
│   ├── __init__.py
│   ├── data_models.py              # FrameData, PlayerData, BallData, CourtData
│   ├── data_collector.py           # Data aggregation and post-processing
│   ├── validators.py               # Data quality validation
│   └── post_processors.py          # Temporal smoothing and interpolation
├── analysis/
│   ├── __init__.py
│   ├── coaching_analyzer.py        # Match analysis and insights
│   ├── movement_analyzer.py        # Speed, distance, coverage metrics
│   ├── rally_analyzer.py           # Rally segmentation and analysis
│   └── metrics.py                  # Utility metric functions
├── video_io/
│   ├── __init__.py
│   ├── video_reader.py             # Video reading and metadata extraction
│   └── video_writer.py             # Video writing
├── config.py                       # Pipeline configuration
├── pipeline.py                     # High-level Pipeline API
├── main.py                         # Main entry point
└── README.md                       # This file
```

## Sub-Pipelines

The orchestrator manages these sub-pipelines (imported from parent directory):
- **CourtCalibrator**: Detects court keypoints and computes homographies
- **PlayerTracker**: Tracks players with bounding boxes and pose estimation
- **BallTracker**: Detects and tracks the ball
- **RallyStateDetector**: Determines rally state (Rally/Not Rally)
