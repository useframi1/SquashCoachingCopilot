# Ball Tracking Tests

This directory contains evaluation scripts for testing the ball tracking pipeline.

## Structure

```
tests/
├── config.json          # Test configuration (video paths, output settings, hit detection params)
├── evaluator.py         # Main evaluation script
├── data/                # Test videos and ground truth data
└── outputs/             # Generated outputs (videos, plots, metrics)
```

## Usage

1. Place your test video in the `data/` directory
2. Update `config.json` with your video path and settings
3. Run the evaluator:

```bash
cd tests
python evaluator.py
```

## Configuration

The `config.json` file contains evaluation-specific settings:

- **video**: Input video path and frame limits
- **tracking**: Visualization settings (trace length, colors, thickness)
- **wall_hit_detection**: Parameters for detecting wall hits
- **racket_hit_detection**: Parameters for detecting racket hits
- **output**: Output paths and formats

The tracker configuration (model selection, preprocessing, postprocessing) is managed separately in `ball_tracking/config.json`.

## Outputs

The evaluator generates:
- Annotated video with ball trajectory and hit markers
- Position and velocity plots
- Hit detection plots (wall and racket hits)
- Performance metrics (CSV)
