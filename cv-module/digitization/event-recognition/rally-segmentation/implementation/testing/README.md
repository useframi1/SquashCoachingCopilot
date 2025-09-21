# Manual Annotation Pipeline for Rally Segmentation

This pipeline allows you to manually annotate rally states while automatically calculating player distance and intensity metrics for threshold analysis.

## Overview

The annotation pipeline helps you:
1. Manually mark rally states (start, active, end) in videos
2. Automatically calculate metrics for the previous N frames (configurable window)
3. Export annotations with calculated statistics to CSV for analysis

## Files

- `manual_annotation_pipeline.py` - Main pipeline implementation
- `run_annotation.py` - Example usage script
- `videos/` - Directory for test videos (place your .mp4 files here)
- `annotations/` - Output directory for CSV files (created automatically)

## Setup

1. Place your test videos in `testing/videos/`
2. Ensure the main pipeline models are available in `pipeline/models/`
3. Make sure you're in the implementation directory when running

## Usage

### Basic Usage
```bash
python testing/run_annotation.py testing/videos/your_video.mp4
```

### With Custom Window Size
```bash
python testing/run_annotation.py testing/videos/your_video.mp4 --window-size 30
```

### Direct Pipeline Usage
```bash
python testing/manual_annotation_pipeline.py testing/videos/your_video.mp4 --window-size 50
```

## Controls

| Key | Action |
|-----|--------|
| `s` | Mark current frame as START state |
| `a` | Mark current frame as ACTIVE state |
| `e` | Mark current frame as END state |
| `SPACE` | Pause/Resume video playback |
| `←` | Seek backward 1 second |
| `→` | Seek forward 1 second |
| `↑` | Seek backward 10 seconds |
| `↓` | Seek forward 10 seconds |
| `q` | Quit and save annotations |

## How It Works

1. **Video Loading**: Opens the specified video file
2. **Player Tracking**: Uses existing PlayerTracker to detect and track players
3. **Metrics Calculation**: For each frame, calculates:
   - Distance between players (in real-world coordinates)
   - Individual player intensity (movement over recent frames)
   - Combined player intensity
4. **Window Analysis**: When you annotate a state, calculates mean/median of metrics over the previous N frames
5. **CSV Export**: Saves all annotations with calculated statistics

## Output Format

The CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `video_name` | Name of the video file |
| `frame_number` | Frame where annotation was made |
| `state` | Annotated state (start/active/end) |
| `window_size` | Number of frames used for calculation |
| `mean_distance` | Mean distance between players |
| `median_distance` | Median distance between players |
| `mean_player1_intensity` | Mean movement intensity of player 1 |
| `median_player1_intensity` | Median movement intensity of player 1 |
| `mean_player2_intensity` | Mean movement intensity of player 2 |
| `median_player2_intensity` | Median movement intensity of player 2 |
| `mean_combined_intensity` | Mean combined player intensity |
| `median_combined_intensity` | Median combined player intensity |

## Configuration

The pipeline uses the main `pipeline/config.json` for:
- Player detection model settings
- Court calibration parameters
- Processing parameters

Key settings:
- `window_size`: Number of frames to analyze (default: 50)
- Player detection confidence thresholds
- Court boundaries for real-world coordinate conversion

## Analysis Workflow

1. **Prepare Videos**: Place test videos in `testing/videos/`
2. **Annotate**: Run the pipeline and mark rally states
3. **Analyze Results**: Use the CSV output to determine optimal thresholds
4. **Iterate**: Test different window sizes and annotation strategies

## Output Files

Annotation files are saved as:
```
testing/annotations/{video_name}_annotations_{timestamp}.csv
```

Example: `rally_sample_annotations_20240921_143022.csv`

## Tips

- Start with a smaller window size (20-30 frames) for responsive feedback
- Pause the video when making annotations for accuracy
- Mark transitions clearly (e.g., when rally starts, becomes active, ends)
- Use seeking controls to review and refine annotations
- Save frequently by pressing 'q' and restarting if needed

## Troubleshooting

**Video won't load**: Check file path and format (MP4 recommended)
**No player detection**: Ensure models are in `pipeline/models/`
**Config errors**: Verify `pipeline/config.json` exists and is valid
**Permission errors**: Check write access to `testing/annotations/` directory