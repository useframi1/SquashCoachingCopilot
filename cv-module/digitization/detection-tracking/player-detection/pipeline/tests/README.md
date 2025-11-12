# Player Tracking Tests

This directory contains evaluation and testing tools for the player tracking pipeline. **Note**: This directory is not included in the package distribution and is only for development purposes.

## Structure

```
tests/
├── evaluator.py           # Main evaluation script
├── config.json            # Test configuration
├── data/                  # Test data (COCO annotations, videos)
│   ├── _annotations.coco.json
│   └── README.md
└── outputs/               # Evaluation outputs (videos, results)
```

## Setup

1. **Install the package in development mode:**
   ```bash
   cd ../  # Go to pipeline directory
   pip install -e .
   ```

2. **Install required dependencies:**
   ```bash
   pip install court_calibration  # For homography computation
   pip install scipy              # For evaluation metrics
   ```

3. **Prepare test data:**
   - Place your test video in `data/clip2.mp4`
   - Ensure COCO annotations match your video frames

## Running Evaluations

### Basic Usage

```bash
python evaluator.py
```

### Custom Configuration

```python
from evaluator import PlayerTrackerEvaluator

# Load custom config
evaluator = PlayerTrackerEvaluator(config=your_config_dict)
results = evaluator.run_evaluation()
```

## Configuration

Edit `config.json` to customize:

- **paths**: Input video, annotations, output locations
- **court_calibration**: Configuration for CourtCalibrator
- **tracker**: Override default tracker configuration (leave `null` to use package defaults)
- **evaluation**: IOU thresholds, player class IDs
- **processing**: Max frames, progress intervals
- **visualization**: Display settings, colors
- **output**: Video codec, result precision

## Output

The evaluator generates:

1. **Console output**: Real-time tracking metrics
2. **Video output**: Annotated video with tracking visualizations (`outputs/tracking_output.mp4`)
3. **Results file**: Detailed metrics in text format (`outputs/tracking_results.txt`)

### Metrics Reported

- **Overall Performance**: Precision, Recall, F1-Score, MOTA
- **Per-Player Performance**: Individual metrics for each player
- **Detection Statistics**: True positives, false positives, false negatives
- **ID Mapping**: Tracker ID to ground truth player mapping

## Dependencies

The evaluator requires:
- `player_tracking` (this package)
- `court_calibration` (for homography computation)
- `scipy` (for Hungarian algorithm in assignment)
- `opencv-python`, `numpy` (already in package dependencies)

## Notes

- The evaluator uses `CourtCalibrator` from the `court_calibration` package to compute the homography matrix from the first frame
- Test data files are not distributed with the package
- The `tests/` directory is excluded from package installation via `pyproject.toml`
