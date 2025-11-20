# Squash Video Analysis Pipeline

Complete end-to-end pipeline for processing squash videos through all 7 stages of analysis.

## Overview

This pipeline orchestrates all modules of the SquashCoachingCopilot system to provide comprehensive squash video analysis:

1. **Court Calibration** - Detect court elements and compute coordinate transformations
2. **Frame-by-Frame Tracking** - Track players and ball throughout the video
3. **Trajectory Postprocessing** - Smooth and interpolate tracking data
4. **Rally Segmentation** - Identify individual rally boundaries
5. **Hit Detection** - Detect wall hits and racket hits
6. **Stroke & Shot Classification** - Classify forehand/backhand strokes and shot types
7. **Export & Visualization** - Generate CSV, annotated video, and statistics

## Quick Start

### Using the CLI

```bash
# Basic usage
python run_pipeline.py --video squashcopilot/pipeline/tests/match.mp4

# Custom configuration
python run_pipeline.py --video match.mp4 --config custom_pipeline.yaml

# Skip video rendering (faster)
python run_pipeline.py --video match.mp4 --no-video

# Custom output directory
python run_pipeline.py --video match.mp4 --output /path/to/results/
```

### Using Python API

```python
from squashcopilot.pipeline import Pipeline

# Initialize with default configuration
pipeline = Pipeline()

# Process video
output_paths = pipeline.run("squashcopilot/pipeline/tests/match.mp4")

# Access outputs
print(f"CSV: {output_paths['csv']}")
print(f"Video: {output_paths['video']}")
print(f"Stats: {output_paths['stats']}")
```

### Using Custom Configuration

```python
from squashcopilot.pipeline import Pipeline

# Load custom config
pipeline = Pipeline(config_path="my_config.yaml")

# Process with custom settings
output_paths = pipeline.run("match.mp4", output_dir="my_results/")
```

## Directory Structure

```
squashcopilot/pipeline/
├── __init__.py              # Pipeline module exports
├── pipeline.py              # Main Pipeline orchestrator class
├── README.md                # This file
├── tests/                   # Test videos (not committed to git)
│   └── .gitkeep
└── outputs/                 # Pipeline outputs
    └── {video-name}/        # Per-video output directory
        ├── {video-name}_analysis.csv
        ├── {video-name}_annotated.mp4
        └── {video-name}_stats.json
```

## Outputs

### 1. CSV Analysis File (`{video_name}_analysis.csv`)

Comprehensive frame-by-frame data with the following columns:

**Frame Metadata:**
- `frame` - Frame number
- `timestamp` - Timestamp in seconds

**Player Data (per player):**
- `player_{N}_x_pixel`, `player_{N}_y_pixel` - Player position in pixels
- `player_{N}_x_meter`, `player_{N}_y_meter` - Player position in meters
- `player_{N}_kp_{keypoint}_x`, `player_{N}_kp_{keypoint}_y` - 12 body keypoints (shoulders, elbows, wrists, hips, knees, ankles)

**Ball Data:**
- `ball_x_pixel`, `ball_y_pixel` - Ball position in pixels
- `ball_x_meter`, `ball_y_meter` - Ball position in meters
- `ball_confidence` - Detection confidence score

**Rally Data:**
- `rally_id` - Which rally this frame belongs to
- `rally_state` - START, PLAY, or END

**Hit Events:**
- `is_wall_hit` - 1 if wall hit on this frame, else 0
- `wall_hit_x`, `wall_hit_y` - Wall hit position
- `is_racket_hit` - 1 if racket hit on this frame, else 0
- `racket_hit_player_id` - Player who hit the ball (1 or 2)
- `racket_hit_x`, `racket_hit_y` - Racket hit position

**Stroke Classification:**
- `stroke_type` - FOREHAND or BACKHAND
- `stroke_confidence` - Classification confidence

**Shot Classification:**
- `shot_type` - STRAIGHT_DROP, STRAIGHT_DRIVE, CROSS_COURT_DROP, or CROSS_COURT_DRIVE
- `shot_direction` - STRAIGHT or CROSS_COURT
- `shot_depth` - DROP or DRIVE
- `shot_confidence` - Classification confidence

### 2. Annotated Video (`{video_name}_annotated.mp4`)

Visual overlay video with:
- Player bounding boxes and IDs
- Player keypoints (skeleton visualization)
- Player trajectory trails
- Ball position marker
- Ball trajectory trail
- Rally ID display
- Stroke type labels on racket hits
- Shot type labels on completed shots

### 3. Statistics JSON (`{video_name}_stats.json`)

High-level analytics including:

```json
{
  "video_info": {
    "filename": "match",
    "duration_seconds": 300.5,
    "fps": 30,
    "total_frames": 9015,
    "resolution": [1920, 1080]
  },
  "rallies": {
    "total_rallies": 45,
    "avg_duration_seconds": 6.7,
    "total_play_time_seconds": 301.5,
    "longest_rally": {"rally_id": 12, "duration": 18.3},
    "shortest_rally": {"rally_id": 3, "duration": 2.1}
  },
  "shots": {
    "total_shots": 234,
    "by_type": {
      "straight_drive": 102,
      "cross_court_drive": 78,
      "straight_drop": 34,
      "cross_court_drop": 20
    },
    "by_direction": {"straight": 136, "cross_court": 98},
    "by_depth": {"drop": 54, "drive": 180}
  },
  "strokes": {
    "player_1": {
      "total": 217,
      "forehand": 120,
      "backhand": 97,
      "forehand_pct": 55.3
    },
    "player_2": {
      "total": 217,
      "forehand": 105,
      "backhand": 112,
      "forehand_pct": 48.4
    }
  },
  "hits": {
    "wall_hits": 468,
    "racket_hits": 234
  }
}
```

## Configuration

The pipeline is configured via `squashcopilot/configs/pipeline.yaml`. Key settings:

### Output Settings

```yaml
output:
  base_directory: "squashcopilot/pipeline/outputs"
  create_video_subdirectory: true
  save_annotated_video: true
  save_csv: true
  save_statistics: true
```

### Module Configuration

Each module can be enabled/disabled and references its own config file:

```yaml
modules:
  court_calibration:
    config_path: "configs/court_calibration.yaml"
    enabled: true
  player_tracking:
    config_path: "configs/player_tracking.yaml"
    enabled: true
  # ... etc
```

### Processing Options

```yaml
processing:
  parallel_tracking: true      # Process player/ball in parallel
  cache_intermediate: false    # Cache intermediate results
  video_writer_codec: "mp4v"   # Video codec
  frame_skip: 1                # Process every Nth frame
```

### Visualization Settings

Customize what appears in the annotated video:

```yaml
visualization:
  draw_court_lines: true
  draw_player_boxes: true
  draw_player_keypoints: true
  draw_player_trajectories: true
  draw_ball: true
  draw_ball_trajectory: true
  show_rally_id: true
  show_stroke_labels: true
  show_shot_labels: true
  trajectory_length: 30  # Number of frames in trajectory trail
```

### Logging Configuration

```yaml
logging:
  level: "INFO"           # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: null          # Optional log file path
  show_progress: true     # Show progress bars
  log_timings: true       # Log stage timing information
```

## CLI Options

```
usage: run_pipeline.py [-h] --video VIDEO [--output OUTPUT] [--config CONFIG]
                       [--no-video] [--no-csv] [--no-stats] [--verbose] [--quiet]

Required arguments:
  --video, -v PATH      Path to input video file

Optional arguments:
  --output, -o DIR      Output directory (default: outputs/{video_name}/)
  --config, -c PATH     Custom pipeline config YAML
  --no-video            Skip annotated video rendering (faster)
  --no-csv              Skip CSV export
  --no-stats            Skip statistics JSON export
  --verbose             Enable DEBUG logging
  --quiet               Suppress progress bars
```

## Examples

### Process Multiple Videos

```bash
# Process all videos in tests/ directory
for video in squashcopilot/pipeline/tests/*.mp4; do
    python run_pipeline.py --video "$video"
done
```

### Quick Analysis (CSV and Stats Only)

```bash
# Skip video rendering for faster processing
python run_pipeline.py --video match.mp4 --no-video
```

### Custom Configuration

```bash
# Use custom settings
python run_pipeline.py --video match.mp4 --config my_pipeline_config.yaml
```

### Batch Processing Script

```python
from pathlib import Path
from squashcopilot.pipeline import Pipeline

# Initialize pipeline once
pipeline = Pipeline()

# Process multiple videos
videos = Path("squashcopilot/pipeline/tests").glob("*.mp4")
for video in videos:
    print(f"Processing {video.name}...")
    try:
        outputs = pipeline.run(str(video))
        print(f"  ✓ Complete: {outputs['csv']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
```

## Performance

Typical processing times on NVIDIA RTX 3090:

| Stage | Time (5min video) | Bottleneck |
|-------|-------------------|------------|
| Court Calibration | ~1s | Roboflow API |
| Tracking | ~12-20s | YOLO + TrackNet inference |
| Postprocessing | <1s | Fast (CPU) |
| Rally Segmentation | ~1s | LSTM inference |
| Hit Detection | ~2s | Signal processing |
| Classification | ~2s | LSTM + rules |
| Video Rendering | ~10s | Video encoding |
| **Total** | **~30-40s** | GPU inference |

CPU-only processing: ~2-3x slower (60-120s for 5min video)

## Troubleshooting

### Court Calibration Fails

**Error**: `Court calibration failed - check API key`

**Solution**: Ensure you have a valid Roboflow API key configured in `configs/court_calibration.yaml`

### Out of Memory

**Error**: `CUDA out of memory`

**Solution**: Reduce batch size in module configs or use CPU-only mode

### Missing Dependencies

**Error**: `ModuleNotFoundError`

**Solution**: Install required dependencies:
```bash
pip install -e .
```

### Video Codec Issues

**Error**: `Could not write video`

**Solution**: Try different codec in config:
```yaml
processing:
  video_writer_codec: "avc1"  # or "mp4v", "xvid"
```

## Advanced Usage

### Disable Specific Modules

```python
from squashcopilot.pipeline import Pipeline

pipeline = Pipeline()

# Disable rally segmentation
pipeline.config["modules"]["rally_state_detection"]["enabled"] = False

# Disable stroke detection
pipeline.config["modules"]["stroke_detection"]["enabled"] = False

pipeline.run("match.mp4")
```

### Custom Visualization Colors

```python
pipeline = Pipeline()

# Change player box color to red
pipeline.config["visualization"]["player_box_color"] = [255, 0, 0]

# Change ball color to green
pipeline.config["visualization"]["ball_color"] = [0, 255, 0]

pipeline.run("match.mp4")
```

### Access Intermediate Results

```python
pipeline = Pipeline()

# Enable intermediate caching
pipeline.config["processing"]["cache_intermediate"] = True

outputs = pipeline.run("match.mp4")

# Access cached data (if implemented)
# ... custom processing on intermediate results
```

## See Also

- [Main README](../../README.md) - Overall project documentation
- [Court Calibration](../modules/court_calibration/README.md)
- [Player Tracking](../modules/player_tracking/README.md)
- [Ball Tracking](../modules/ball_tracking/README.md)
- [Rally Detection](../modules/rally_state_detection/README.md)
- [Stroke Detection](../modules/stroke_detection/README.md)
- [Shot Classification](../modules/shot_type_classification/README.md)
