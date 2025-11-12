# Annotation Pipeline

This directory contains the manual annotation pipeline for creating training data for rally state detection.

## Overview

The annotation pipeline allows you to manually label rally states (start, active, end) while the system automatically:
- Tracks players using the `player_tracking` package
- Calibrates the court using the `court_calibration` package
- Calculates frame-by-frame metrics (player positions and distances)
- Saves all data to CSV files

## Requirements

Make sure you have the following packages installed:
```bash
pip install player_tracking
pip install court_calibration
pip install rally_state_detection
```

## Usage

### Basic Usage

Run the annotation pipeline with default config:
```bash
python annotation_pipeline.py --video path/to/video.mp4
```

### Using Custom Config

Create a config file with your settings and run:
```bash
python annotation_pipeline.py --video path/to/video.mp4 --config ../config.json
```

### Config Structure

The config file should include an `annotations` section:
```json
{
    "annotations": {
        "video_path": "videos/video-2.mp4",
        "data_path": "data"
    }
}
```

## Controls

While the annotation pipeline is running:

- **s**: Mark current frame as START state
- **a**: Mark current frame as ACTIVE state
- **e**: Mark current frame as END state
- **SPACE**: Pause/Resume video
- **q**: Quit and save annotations
- **LEFT/RIGHT arrows**: Seek backward/forward by 1 second
- **UP/DOWN arrows**: Seek backward/forward by 10 seconds

## Output

The pipeline saves frame-by-frame annotations to CSV files with the following columns:

- `frame`: Frame number
- `timestamp`: Time in seconds
- `state`: Current rally state (start, active, end)
- `video_name`: Name of the video
- `player_distance`: Distance between players in meters
- `player1_x`, `player1_y`: Player 1 real-world coordinates
- `player2_x`, `player2_y`: Player 2 real-world coordinates

Files are saved to `{data_path}/{video_name}/{video_name}_annotations_{timestamp}.csv`

## Key Differences from Implementation Version

This pipeline version differs from the implementation version in several ways:

1. **Installed Packages**: Uses the installed `player_tracking` and `court_calibration` packages instead of local implementations
2. **Metrics Aggregator**: Uses the correct metrics aggregator from `rally_state_detection.utilities.metrics_aggregator`
3. **Frame-by-Frame Metrics**: Saves metrics for every frame without aggregation, allowing for more granular analysis
4. **No Auto-save Counter**: Removed the window-based auto-save mechanism since we save every frame

## Troubleshooting

### Court Calibration Fails
- Ensure the first frame of your video shows a clear view of the court
- Check that your Roboflow API key is properly configured in the court calibration package

### Player Tracking Issues
- The tracker initializes players based on left/right positioning in the first frame
- If tracking is lost, the system will attempt to re-initialize

### Missing Dependencies
- Ensure all required packages are installed: `player_tracking`, `court_calibration`, `rally_state_detection`
- Install additional dependencies: `opencv-python`, `pandas`, `numpy`
