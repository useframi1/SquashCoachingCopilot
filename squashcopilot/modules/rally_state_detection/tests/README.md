# Rally State Detection - Testing Suite

This directory contains the testing and evaluation infrastructure for the rally state detection package.

## Directory Structure

```
tests/
├── config.json                 # Test configuration
├── evaluator.py               # Model evaluation script
├── train_model.py             # Model training script
├── data/                      # Test data organized by video
│   ├── video-1/
│   │   ├── video-1.csv       # Annotations
│   │   └── video-1.mp4       # Video file (optional)
│   ├── video-2/
│   │   ├── video-2.csv
│   │   └── video-2.mp4
│   └── ...
└── outputs/                   # Evaluation results
    └── {video_name}/
        ├── predictions_plot.png
        └── metrics.txt
```

## Configuration

Edit `config.json` to configure testing parameters:

```json
{
    "data": {
        "data_dir": "data",
        "video_name": "video-2"  # Which video to evaluate on
    },
    "evaluation": {
        "tolerance_frames": 2     # Frame tolerance for evaluation
    },
    "output": {
        "output_dir": "outputs",
        "plot_dpi": 300
    }
}
```

## Training the Model

To train a new model on your annotated data:

```bash
cd tests
python train_model.py
```

This will:
1. Load all annotation CSVs from the `data/` directory
2. Engineer features for rally state prediction
3. Split data by video (avoiding data leakage)
4. Train the XGBoost model
5. Evaluate on test videos
6. Save the trained model to `rally_state_detection/models/ml/weights/`

## Running Evaluation

To evaluate the model on a specific video:

1. Update `config.json` with the desired video name
2. Run the evaluator:

```bash
cd tests
python evaluator.py
```

This will:
1. Load the annotation data for the specified video
2. Run rally state detection (raw predictions)
3. Apply postprocessing (minimum duration filtering)
4. Calculate metrics (strict and with tolerance)
5. Generate visualization plots
6. Save results to `outputs/{video_name}/`

## Output Files

### metrics.txt
Contains detailed evaluation metrics:
- **Raw Predictions**: Accuracy before postprocessing
- **Postprocessed (Strict)**: Accuracy after postprocessing, no tolerance
- **With Tolerance**: Accuracy with ±N frames tolerance
- Per-class metrics (precision, recall, F1-score)
- Confusion matrices

### predictions_plot.png
Visualization showing three plots:
1. **Ground Truth States**: Annotated states
2. **Raw Predicted States**: Model predictions before postprocessing
3. **Postprocessed States**: Final predictions after minimum duration filtering

## Data Format

Annotation CSV files should contain the following columns:

### Required Columns:
- `frame_number`: Frame index
- `mean_distance` (or `player_distance`): Distance between players (meters)
- `median_player1_x` (or `player1_x`): Player 1 X coordinate (meters)
- `median_player1_y` (or `player1_y`): Player 1 Y coordinate (meters)
- `median_player2_x` (or `player2_x`): Player 2 X coordinate (meters)
- `median_player2_y` (or `player2_y`): Player 2 Y coordinate (meters)
- `state`: Ground truth rally state ("start", "active", or "end")

### Optional Columns:
- `video_name`: Video identifier (auto-generated from filename if missing)

## Example Workflow

1. **Prepare Data**:
   ```bash
   mkdir -p data/my-video
   # Place my-video.csv in data/my-video/
   ```

2. **Train Model** (if needed):
   ```bash
   python train_model.py
   ```

3. **Evaluate**:
   ```bash
   # Edit config.json: set video_name to "my-video"
   python evaluator.py
   ```

4. **Review Results**:
   ```bash
   cat outputs/my-video/metrics.txt
   open outputs/my-video/predictions_plot.png
   ```

## Notes

- This test suite is excluded from the package distribution
- Training requires multiple annotated videos for proper cross-validation
- The evaluator uses the package's `RallyStateDetector` for batch processing
- Postprocessing enforces minimum state durations and valid state transitions
