# Quick Start Guide

## Prerequisites

Ensure you have the following sub-pipelines in the parent directory:
- `court_detection_pipeline.py` (CourtCalibrator)
- `player_tracking_pipeline.py` (PlayerTracker)
- `ball_detection_pipeline.py` (BallTracker)
- `rally_state_pipeline.py` (RallyStateDetector)

Required dependencies:
- OpenCV (`cv2`)
- NumPy
- YOLO models for player tracking
- Other dependencies from sub-pipelines

## Quick Start

### Option 1: Using main.py (Simplest)

1. Edit [config.py](config.py) to set your video path:
```python
@dataclass
class PipelineConfig:
    video_path: str = "your_video.mp4"  # Change this
    output_path: str = "output_video.mp4"
    analysis_output_path: str = "match_analysis"
    display: bool = True
```

2. Run:
```bash
python main.py
```

This will:
- Process the video through all sub-pipelines
- Generate an annotated output video
- Create `match_analysis.json` and `match_analysis.txt` with insights

### Option 2: Using Pipeline API

```python
from pipeline import Pipeline
from config import PipelineConfig

# Configure
config = PipelineConfig(
    video_path="video.mp4",
    output_path="output.mp4",
    analysis_output_path="match_analysis",
    display=True
)

# Run pipeline
pipeline = Pipeline(config=config)
analysis = pipeline.run()

print(f"Total rallies: {analysis['rally_statistics']['total_rallies']}")
```

### Option 3: Advanced (Direct Component Access)

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
    handle_missing_data=True
)

visualizer = Visualizer(
    show_player_keypoints=True,
    keypoint_confidence_threshold=0.5
)

orchestrator = PipelineOrchestrator(
    data_collector=data_collector,
    visualizer=visualizer
)

# 2. Process video
with VideoReader("video.mp4") as reader:
    metadata = reader.get_metadata()
    print(f"Processing: {metadata}")

    with VideoWriter("output.mp4", metadata) as writer:
        orchestrator.process_frames(
            frames_iterator=reader.frames(),
            video_metadata={
                "fps": metadata.fps,
                "width": metadata.width,
                "height": metadata.height,
                "total_frames": metadata.total_frames
            },
            display=True,
            on_frame_processed=lambda fd, frame: writer.write(frame)
        )

# 3. Analyze
frames = orchestrator.get_collected_data()
analyzer = CoachingAnalyzer(fps=metadata.fps)
match_analysis = analyzer.analyze_match(frames)

# 4. Export
analyzer.export_analysis(frames, "analysis.json", format="json")
analyzer.export_analysis(frames, "summary.txt", format="summary")
```

## Available Analyses

### Match Analysis
```python
analysis = analyzer.analyze_match(frames)
# Returns:
# - match_info (duration, frames)
# - movement_analysis (both players)
# - rally_statistics
# - rallies (list of individual rallies)
# - coaching_insights
```

### Player Performance
```python
player_analysis = analyzer.analyze_player_performance(frames, player_id=1)
# Returns:
# - movement_metrics
# - speed_profile_summary
# - acceleration_profile_summary
# - sprints (count and details)
# - positioning (coverage, variability)
```

### Player Comparison
```python
comparison = analyzer.compare_players(frames)
# Returns:
# - player1 (full analysis)
# - player2 (full analysis)
# - comparison_summary (who's better at what)
```

## Configuration Options

### DataCollector
```python
DataCollector(
    enable_smoothing=True,          # Apply temporal smoothing
    smoothing_window=5,             # Window size for smoothing
    enable_validation=True,         # Validate data quality
    min_confidence=0.3,             # Min confidence threshold
    max_position_change=200.0,      # Max pixel change per frame
    handle_missing_data=True,       # Interpolate missing data
    max_interpolation_frames=10     # Max frames to interpolate
)
```

### Visualizer
```python
Visualizer(
    show_court_keypoints=True,           # Show court calibration
    show_player_keypoints=True,          # Show player pose
    show_player_bbox=True,               # Show bounding boxes
    show_ball=True,                      # Show ball position
    show_rally_state=True,               # Show rally state
    keypoint_confidence_threshold=0.5    # Min confidence for keypoints
)
```

## Running the Pipeline

```bash
# Make sure you're in the full-pipeline directory
cd /path/to/cv-module/full-pipeline

# Run the pipeline
python main.py
```

The pipeline will:
1. Read video frames using VideoReader
2. Calibrate court on first frame
3. Track players with pose estimation
4. Detect ball position
5. Determine rally state
6. Collect and validate data
7. Apply post-processing (smoothing, interpolation)
8. Generate annotated output video
9. Analyze match and export results

## Output Files

### JSON Export
Complete machine-readable analysis:
```json
{
  "match_info": {...},
  "movement_analysis": {...},
  "rally_statistics": {...},
  "rallies": [...],
  "coaching_insights": [...]
}
```

### Text Summary
Human-readable coaching report:
```
============================================================
SQUASH MATCH ANALYSIS SUMMARY
============================================================

MATCH INFORMATION
------------------------------------------------------------
Duration: 150.0 seconds
Total Frames: 4500

RALLY STATISTICS
------------------------------------------------------------
Total Rallies: 12
Average Duration: 8.5s
...
```

## Key Metrics Explained

### Movement Metrics
- **total_distance**: Total meters traveled
- **average_speed**: Average speed in m/s
- **max_speed**: Peak speed reached
- **court_coverage**: Percentage of court visited (0-100%)
- **direction_changes**: Number of significant direction changes

### Rally Metrics
- **duration**: Rally length in seconds
- **intensity**: Average speed during rally
- **shot_count**: Estimated number of shots

### Positioning Metrics
- **average_position**: Center of activity on court
- **positioning_variability**: How much player moves around

## Next Steps

1. Run `test.py` to verify everything works
2. Run `test_advanced.py` to see full capabilities
3. Check `ARCHITECTURE.md` for design details
4. Modify configuration to suit your needs
5. Build custom analyzers by extending base classes
