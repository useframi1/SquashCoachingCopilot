# Shot Type Classification Tests

This directory contains tests for the shot type classification package.

## Running Tests

### Unit Tests

Run the unit tests with:

```bash
python -m pytest tests/test_shot_classifier.py -v
```

Or using unittest:

```bash
python -m unittest tests/test_shot_classifier.py
```

### Shot Classification Evaluator

The evaluator reads ball tracking data from CSV and classifies shots.

**Prerequisites:**
1. Run the ball tracking evaluator first to generate the CSV file:
   ```bash
   cd ../../ball-detection/pipeline/tests
   python evaluator.py
   ```
   This generates: `outputs/{video_name}/postprocessed/{video_name}_ball_positions.csv`

2. Set up the data directory structure:
   ```bash
   cd ../../../shot-type-classification/pipeline/tests
   mkdir -p data/video-3
   ```

3. Copy the CSV and video to the data directory:
   ```bash
   # Copy CSV from ball tracking outputs
   cp ../../ball-detection/pipeline/tests/outputs/video-3/postprocessed/video-3_ball_positions.csv \
      data/video-3/

   # Copy or link the video file
   cp ../../ball-detection/pipeline/tests/data/video-3.mp4 data/video-3/
   ```

4. Run the shot classification evaluator:
   ```bash
   python evaluator.py
   ```

**Configuration:**

Edit `config.json` to specify:
- `test_video`: Video name (e.g., "video-3", "video-1")
- `tracking`: Ball trace visualization settings
- `output`: Video and plot generation settings

**Data Directory Structure:**
```
tests/
├── data/
│   └── {video-name}/
│       ├── {video-name}_ball_positions.csv
│       └── {video-name}.mp4
└── outputs/
    └── {video-name}/
        ├── trajectory_plots.png
        ├── annotated.mp4
        └── report.txt
```

**Output:**

The evaluator generates:
- `trajectory_plots.png`: Trajectory analysis with shot classifications
- `annotated.mp4`: Video with shot type overlays
- `report.txt`: Detailed shot-by-shot breakdown

## Test Coverage

The test suite covers:

- **Shot Classification**: Direction and depth classification
- **Feature Extraction**: Velocity, distance, and trajectory features
- **Shot Types**: Enum definitions and data structures
- **Statistics**: Shot statistics calculation

## Adding New Tests

When adding new functionality, please add corresponding unit tests to ensure reliability and prevent regressions.
