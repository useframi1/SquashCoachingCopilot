# Shot Type Classification

A package for classifying shot types in squash videos based on trajectory analysis.

## Features

- Shot direction classification (straight, cross-court, down-the-line)
- Shot depth classification (drop, long/drive)
- Combined shot type classification
- Feature extraction from ball trajectories
- Configurable thresholds for classification

## Installation

```bash
pip install -e .
```

## Usage

```python
from shot_type_classification import ShotClassifier

# Initialize classifier
classifier = ShotClassifier(fps=30)

# Classify shots
shots = classifier.classify(
    ball_positions=ball_positions,
    wall_hits=wall_hits,
    racket_hits=racket_hits
)

# Get statistics
stats = classifier.get_statistics(shots)
```

## Configuration

Configuration can be customized via the `config.json` file or by passing a config dictionary to the classifier:

```python
config = {
    "shot_classification": {
        "direction_thresholds": {
            "straight_max_angle_deg": 30,
            "cross_min_angle_deg": 120
        },
        "depth_thresholds": {
            "drop_max_rebound_distance_px": 400
        }
    }
}

classifier = ShotClassifier(config=config)
```

## Shot Types

The classifier recognizes the following shot types:

- **Straight Drive**: Long shot hit straight
- **Straight Drop**: Short shot hit straight
- **Cross Court Drive**: Long shot hit diagonally across the court
- **Cross Court Drop**: Short shot hit diagonally across the court
- **Down-the-Line Drive**: Long shot hit along the side wall
- **Down-the-Line Drop**: Short shot hit along the side wall

## Requirements

- numpy
- scipy
