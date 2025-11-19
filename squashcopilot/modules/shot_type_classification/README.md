# Shot Type Classification Module

The shot type classification module analyzes squash shots by classifying their direction and depth based on ball trajectory, wall hits, and player positions.

## Overview

This module provides rule-based shot classification for squash videos by:

-   **Direction Classification**: Determines if shot is STRAIGHT or CROSS_COURT
-   **Depth Classification**: Determines if shot is DROP (short) or LONG (drive)
-   **Combined Classification**: Generates complete shot types (e.g., STRAIGHT_DROP, CROSS_COURT_DRIVE)
-   **Feature Extraction**: Analyzes racket hit positions, wall hit positions, and rebound distances
-   **Statistics Generation**: Computes shot distribution and success rates

The module is part of the SquashCoachingCopilot package and provides tactical analysis of player shot selection during rallies.

## Features

- **Rule-based classification** (no ML model required)
- **Direction detection** using court geometry
- **Depth classification** based on rebound distance
- **Shot-by-shot analysis** with detailed features
- **Aggregated statistics** for tactical insights
- **Configurable thresholds** for classification rules
- **Real-world coordinates** for accurate spatial analysis

## Components

### ShotClassifier (`shot_classifier.py`)

The main class for shot classification.

**Key Methods:**
- `classify(input: ShotClassificationInput)`: Classify all shots in a video

**Classification Pipeline:**
1. **Shot Window Definition**: Find consecutive racket hits to define shot boundaries
2. **Wall Hit Matching**: Match wall hit within shot window
3. **Feature Extraction**:
   - Racket hit position (meters)
   - Next racket hit position (meters)
   - Wall hit distance from center
   - Rebound distance (wall hit to next racket hit)
4. **Direction Classification**: Compare positions relative to court center
5. **Depth Classification**: Analyze rebound distance
6. **Confidence Assignment**: 1.0 if wall hit detected, 0.7 otherwise

## Data Models

The module uses standardized data models from `squashcopilot.common.models.shot`:

### Input Models
- **ShotClassificationInput**: Complete shot analysis data
  - `player1_positions_meter`: List of player 1 positions in meters (Point2D objects)
  - `player2_positions_meter`: List of player 2 positions in meters (Point2D objects)
  - `wall_hits`: List of WallHit objects with detected wall hits
  - `racket_hits`: List of RacketHit objects with detected racket hits

### Output Models
- **ShotResult**: Single shot classification
  - `frame`: Frame number of racket hit
  - `direction`: ShotDirection (STRAIGHT or CROSS_COURT)
  - `depth`: ShotDepth (DROP or LONG)
  - `shot_type`: ShotType (combined classification)
  - `racket_hit_pos`: Point2D in meters (position where racket hit the ball)
  - `next_racket_hit_pos`: Optional Point2D in meters (position of next racket hit)
  - `wall_hit_pos`: Optional Point2D in meters (position where ball hit wall, if detected)
  - `wall_hit_frame`: Optional int (frame number of wall hit, if detected)
  - `rebound_distance`: Optional float (distance of ball rebound in meters)
  - `confidence`: Classification confidence (0.0-1.0)
  - `has_wall_hit()`: Method to check if shot has an associated wall hit
  - `to_dict()`: Method to convert to dictionary

- **ShotClassificationResult**: Complete classification results
  - `shots`: List of ShotResult objects
  - `num_shots`: Number of shots classified (computed automatically)
  - `wall_hit_detection_rate`: Percentage of shots with detected wall hits (computed automatically)
  - `get_statistics()`: Method to generate ShotStatistics from results
  - `to_dict()`: Method to convert to dictionary

- **ShotStatistics**: Aggregated shot analysis (generated via `get_statistics()`)
  - `total_shots`: Total number of shots
  - `by_type`: Dict mapping shot type names to counts
  - `by_direction`: Dict mapping directions to counts (STRAIGHT, CROSS_COURT)
  - `by_depth`: Dict mapping depths to counts (DROP, LONG)
  - `wall_hit_detection_rate`: Percentage of shots with detected wall hits
  - `average_rebound_distance`: Average rebound distance in meters (optional)
  - `to_dict()`: Method to convert to dictionary

## Shot Types

The classifier recognizes the following shot types (from `ShotType` enum):

- **STRAIGHT_DRIVE**: Long shot hit straight down the wall
- **STRAIGHT_DROP**: Short shot hit straight (low rebound)
- **CROSS_COURT_DRIVE**: Long shot hit diagonally across court
- **CROSS_COURT_DROP**: Short shot hit diagonally (low rebound)

## Usage

### Using Standard Data Models

```python
from squashcopilot import ShotClassifier
from squashcopilot import ShotClassificationInput

# Initialize classifier
classifier = ShotClassifier()

# Prepare input (from annotation results)
input_data = ShotClassificationInput(
    player1_positions_meter=player1_positions_meters,  # List[Point2D]
    player2_positions_meter=player2_positions_meters,  # List[Point2D]
    wall_hits=wall_hit_results.hits,                  # List[WallHit]
    racket_hits=racket_hit_results.hits               # List[RacketHit]
)

# Classify all shots
result = classifier.classify(input_data)

# Access shot details
for shot in result.shots:
    print(f"Frame {shot.frame}")
    print(f"  Type: {shot.shot_type.name}")
    print(f"  Direction: {shot.direction.name}")
    print(f"  Depth: {shot.depth.name}")
    print(f"  Confidence: {shot.confidence:.2f}")
    if shot.rebound_distance:
        print(f"  Rebound distance: {shot.rebound_distance:.2f}m")

# Get statistics
stats = result.get_statistics()
print(f"Total shots: {stats.total_shots}")
print(f"By type: {stats.by_type}")
print(f"By direction: {stats.by_direction}")
print(f"By depth: {stats.by_depth}")
print(f"Wall hit detection rate: {stats.wall_hit_detection_rate:.1%}")
```

### Integration with Other Modules

```python
from squashcopilot import Annotator

# Annotator handles shot classification automatically
annotator = Annotator()
results = annotator.annotate_video("video.mp4", "output_directory")

# Access shot classification results
shot_result = results['shot_classification']
statistics = shot_result.statistics
```

## Configuration

Configuration file: `squashcopilot/configs/shot_type_classification.yaml`

```yaml
court_geometry:
  center_x: 4.57  # Court center X coordinate in meters (width/2 + offset)
  center_y: 4.875  # Court center Y coordinate in meters (length/2)

direction_thresholds:
  straight_tolerance: 1.0  # Max distance from center line (meters)
  cross_court_min_distance: 2.0  # Min distance crossed (meters)

depth_thresholds:
  drop_max_rebound: 3.0  # Max rebound distance for drop shot (meters)
  drive_min_rebound: 5.0  # Min rebound distance for drive (meters)

confidence:
  with_wall_hit: 1.0
  without_wall_hit: 0.7
```

### Configuration Parameters

**Court Geometry:**
- `center_x`: X-coordinate of court center in meters (4.57m)
- `center_y`: Y-coordinate of court center in meters (4.875m)

**Direction Thresholds:**
- `straight_tolerance`: Maximum deviation from center line for straight shot (1.0m)
- `cross_court_min_distance`: Minimum lateral distance for cross-court shot (2.0m)

**Depth Thresholds:**
- `drop_max_rebound`: Maximum rebound distance for drop shot (3.0m)
- `drive_min_rebound`: Minimum rebound distance for drive shot (5.0m)

**Confidence:**
- `with_wall_hit`: Confidence when wall hit is detected (1.0)
- `without_wall_hit`: Confidence when wall hit is inferred (0.7)

## Algorithm Details

### Shot Window Definition
1. Iterate through racket hits in chronological order
2. Define shot window: current racket hit → next racket hit
3. Look for wall hit within this window

### Direction Classification
```
if |racket_hit_x - next_racket_hit_x| < straight_tolerance:
    direction = STRAIGHT
elif |racket_hit_x - next_racket_hit_x| > cross_court_min_distance:
    direction = CROSS_COURT
else:
    direction = STRAIGHT  # Default
```

### Depth Classification
```
rebound_distance = distance(wall_hit, next_racket_hit)

if rebound_distance < drop_max_rebound:
    depth = DROP
elif rebound_distance > drive_min_rebound:
    depth = LONG
else:
    depth = LONG  # Default
```

### Coordinate System
- Origin (0, 0) at back-left corner of court
- X-axis: left (0m) to right (6.4m)
- Y-axis: back wall (0m) to front wall (9.75m)
- Court center: approximately (3.2m, 4.875m)

## Module Structure

```
shot_type_classification/
├── __init__.py                # Package exports
├── shot_classifier.py         # Main ShotClassifier class
└── tests/                     # Evaluation suite
    ├── evaluator.py           # Evaluation script
    ├── data/                  # Test videos
    └── outputs/               # Results and metrics
```

## Testing and Evaluation

The `tests/` directory contains evaluation tools:

- **evaluator.py**: Tests classification accuracy against manual annotations
- **data/**: Test videos with ground truth shot labels
- **outputs/**: Classification results and confusion matrices

### Running Tests

```python
from squashcopilot.modules.shot_type_classification.tests.evaluator import ShotClassificationEvaluator

evaluator = ShotClassificationEvaluator()
results = evaluator.evaluate(video_path="tests/data/video-1.mp4")
evaluator.generate_report(results)
```

## Performance Considerations

- **Classification**: Real-time (>1000 shots/second)
- **No ML inference**: Rule-based, no GPU needed
- **Lightweight**: Minimal memory footprint
- **Scales linearly**: O(n) where n = number of racket hits

## Limitations

- Rule-based approach may not capture nuanced shot variations
- Requires accurate wall hit and racket hit detection
- Classification quality depends on court calibration accuracy
- Fixed thresholds may not generalize across different courts/cameras
- Cannot classify special shots (lobs, boasts, kills) without additional rules
- Assumes standard squash court dimensions

## Dependencies on Other Modules

This module requires data from:
- **Ball Tracking**: Wall hits and racket hits
- **Player Tracking**: Player positions in meters
- **Court Calibration**: Floor homography for coordinate transformation

## Future Improvements

- ML-based classification for more nuanced shot types
- Additional shot categories (lob, boast, kill, volley)
- Player-specific shot pattern analysis
- Shot quality metrics (pace, accuracy, placement)
- Adaptive thresholds based on player style

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

MIT License

This module is part of the SquashCoachingCopilot thesis project at the American University in Cairo.
