# Court Calibration Module

The court calibration module detects squash court features and computes transformation matrices to map between pixel coordinates and real-world court coordinates.

## Overview

This module provides automated court calibration for squash videos by:

-   Detecting court elements (tin, service boxes, T-boxes, and wall markers) using Roboflow API
-   Extracting corner keypoints from detected polygons
-   Computing homography matrices for coordinate transformation (pixel ↔ meters)
-   Detecting wall color to determine ball visibility settings
-   Supporting both floor and wall coordinate systems

The module is part of the SquashCoachingCopilot package and provides essential calibration data used by all other modules for accurate real-world measurements.

## Features

- **Court Element Detection**: Detects tin, service boxes, T-boxes, and wall markers
- **Polygon Processing**: Extracts corner points from detected quadrilaterals
- **Homography Computation**: Creates transformation matrices for floor and wall
- **Wall Color Analysis**: Determines wall color for optimal ball detection settings
- **Roboflow Integration**: Uses Roboflow API for robust court detection
- **Real-world Mapping**: Converts pixel coordinates to meters and vice versa

## Components

### CourtCalibrator (`court_calibrator.py`)

The main class for court detection and calibration.

**Key Methods:**
- `detect_keypoints(frame: Frame)`: Detect court element keypoints
- `process_frame(input: CourtCalibrationInput)`: Full calibration pipeline
- `detect_wall_color(input: WallColorDetectionInput)`: Analyze wall color
- `get_homography(element_name: str)`: Retrieve cached homography matrix

**Detection Pipeline:**
1. **API Call**: Send frame to Roboflow for court element detection
2. **Polygon Extraction**: Extract bounding polygons for each court element
3. **Corner Detection**: Approximate polygons to quadrilaterals and extract corners
4. **Homography Computation**: Calculate transformation matrices using detected and real-world points
5. **Caching**: Store homographies for reuse

### Dependencies

-   **NumPy**: Numerical operations
-   **OpenCV**: Image processing and homography computation
-   **inference**: Roboflow API client for court detection

## Data Models

The module uses standardized data models from `squashcopilot.common.models.court`:

### Input Models
- **CourtCalibrationInput**: Single frame for court detection
  - `frame`: Frame object with image data

- **WallColorDetectionInput**: Frame with wall mask for color analysis
  - `frame`: Frame object
  - `wall_mask`: Binary mask of wall region (optional, auto-detected if not provided)

### Output Models
- **CourtCalibrationResult**: Complete calibration data
  - `floor_homography`: Homography (3x3 matrix) for floor coordinate transformation
  - `wall_homography`: Homography for wall coordinate transformation
  - `keypoints`: Dictionary mapping court elements to their corner Keypoints
    - `tin`: 4 corner points
    - `left_service_box`: 4 corner points
    - `right_service_box`: 4 corner points
    - Other court elements as detected

- **WallColorDetectionResult**: Wall color analysis
  - `is_dark_wall`: Boolean indicating if wall is dark (requires black ball preprocessing)
  - `brightness`: Average brightness value (0-255)
  - `rgb_color`: Tuple (R, G, B) of average wall color
  - `bgr_color`: Tuple (B, G, R) for OpenCV compatibility

- **Keypoints**: Collection of named 2D points
  - Points stored as Point2D objects with x, y coordinates
  - Optional confidence scores per point

- **Homography**: 3x3 transformation matrix
  - `matrix`: NumPy array (3, 3)
  - `transform_point(point)`: Transform single point
  - `transform_points(points)`: Batch transformation
  - `inverse()`: Get inverse homography

## Usage

### Using Standard Data Models

```python
from squashcopilot import CourtCalibrator
from squashcopilot import CourtCalibrationInput, Frame
import cv2

# Initialize calibrator
calibrator = CourtCalibrator()

# Read first frame
cap = cv2.VideoCapture("squash_video.mp4")
ret, frame_img = cap.read()

# Create Frame object
frame = Frame(image=frame_img, frame_number=0, timestamp=0.0)

# Create input
input_data = CourtCalibrationInput(frame=frame)

# Process frame
result = calibrator.process_frame(input_data)

# Access homographies
floor_H = result.floor_homography
wall_H = result.wall_homography

# Transform a pixel point to meters
from squashcopilot import Point2D
pixel_point = Point2D(x=320, y=240)
meter_point = floor_H.transform_point(pixel_point)
print(f"Point in meters: ({meter_point.x:.2f}, {meter_point.y:.2f})")

# Detect wall color
wall_color_result = calibrator.detect_wall_color(
    WallColorDetectionInput(frame=frame)
)
is_black_ball = wall_color_result.is_dark_wall
print(f"Use black ball preprocessing: {is_black_ball}")

cap.release()
```

### Integration with Other Modules

```python
from squashcopilot import Annotator

# Annotator handles court calibration automatically
annotator = Annotator()
results = annotator.annotate_video("video.mp4", "output_directory")

# Court calibration is performed on first frame
# Other modules access homographies as needed
```

## Configuration

Configuration file: `squashcopilot/configs/court_calibration.yaml`

```yaml
roboflow:
  model_id: "squash-court-court-detection/4"
  api_key: "${ROBOFLOW_API_KEY}"  # Set as environment variable

class_mapping:
  tin: "tin"
  left_service_box: "left_square"
  right_service_box: "right_square"
  # Additional court elements...

real_world_coords:
  floor:
    # Coordinates in meters for floor elements
    width: 6.4
    length: 9.75
    service_line: 5.44
    t_line: 7.04

  wall:
    # Coordinates in meters for wall elements
    width: 6.4
    service_line_height: 1.78
    tin_height: 0.48

wall_color_detection:
  brightness_threshold: 128  # Below this is considered dark
```

### Configuration Parameters

**Roboflow:**
- `model_id`: Roboflow model identifier for court detection
- `api_key`: API key for Roboflow (set as environment variable for security)

**Class Mapping:**
- Maps detected class names to standardized element names
- Example: `"tin"` in detection → `tin` in results

**Real-world Coordinates:**
- Standard squash court dimensions in meters
- **Floor dimensions:**
  - Width: 6.4m
  - Length: 9.75m
  - Service line: 5.44m from back wall
  - T-line: 7.04m from back wall
- **Wall dimensions:**
  - Width: 6.4m
  - Service line height: 1.78m
  - Tin height: 0.48m

**Wall Color Detection:**
- `brightness_threshold`: Brightness value (0-255) below which wall is considered dark

## Algorithm Details

### Court Element Detection
1. **Roboflow API Call**: Send frame to Roboflow model for segmentation
2. **Polygon Extraction**: Extract bounding polygons from detection masks
3. **Quadrilateral Approximation**: Use Douglas-Peucker algorithm to approximate polygons as quadrilaterals
4. **Corner Ordering**: Order corners consistently (e.g., top-left, top-right, bottom-right, bottom-left)

### Homography Computation
Homographies map between pixel coordinates (source) and real-world coordinates (destination):

**Floor Homography:**
- Source: Detected floor element corners (pixels)
- Destination: Real-world court coordinates (meters)
- Use: Transform player/ball positions to real-world measurements

**Wall Homography:**
- Source: Detected wall element corners (pixels)
- Destination: Real-world wall coordinates (meters)
- Use: Transform wall hit positions for shot analysis

**Method**: `cv2.findHomography()` computes the 3x3 transformation matrix

### Wall Color Detection
1. **Wall Region Extraction**: Use detected wall polygon as mask
2. **Color Averaging**: Calculate mean RGB/BGR values within wall region
3. **Brightness Calculation**: Convert to grayscale and compute mean brightness
4. **Classification**: Compare brightness to threshold (128) to determine if dark

## API Reference

### CourtCalibrator

#### `__init__(config: Config = None)`
Initialize the court calibrator with optional configuration.

**Parameters:**
- `config` (Config, optional): Configuration object loaded from YAML

#### `detect_keypoints(frame: Frame) -> Dict[str, Keypoints]`
Detect keypoints from court elements.

**Parameters:**
- `frame`: Frame object with image data

**Returns:**
- Dictionary mapping element names to Keypoints objects

#### `process_frame(input: CourtCalibrationInput) -> CourtCalibrationResult`
Compute homography matrices for court calibration.

**Parameters:**
- `input`: CourtCalibrationInput with frame data

**Returns:**
- CourtCalibrationResult with floor/wall homographies and keypoints

#### `detect_wall_color(input: WallColorDetectionInput) -> WallColorDetectionResult`
Analyze wall color to determine ball visibility settings.

**Parameters:**
- `input`: WallColorDetectionInput with frame and optional wall mask

**Returns:**
- WallColorDetectionResult with color analysis

#### `get_homography(element_name: str) -> Homography`
Retrieve a cached homography matrix.

**Parameters:**
- `element_name`: Name of court element ("floor" or "wall")

**Returns:**
- Homography object with transformation matrix

**Raises:**
- `ValueError`: If homography not found (call process_frame first)

## Module Structure

```
court_calibration/
├── __init__.py                # Package exports
├── court_calibrator.py        # Main CourtCalibrator class
└── tests/                     # Evaluation suite
    ├── evaluator.py           # Evaluation script
    ├── data/                  # Test videos
    └── outputs/               # Results and metrics
```

## Testing and Evaluation

The `tests/` directory contains evaluation tools:

- **evaluator.py**: Tests calibration accuracy against manual annotations
- **data/**: Test videos with known court dimensions
- **outputs/**: Calibration visualizations and accuracy metrics

### Running Tests

```python
from squashcopilot.modules.court_calibration.tests.evaluator import CourtCalibrationEvaluator

evaluator = CourtCalibrationEvaluator()
results = evaluator.evaluate(video_path="tests/data/video-1.mp4")
evaluator.generate_report(results)
```

## Performance Considerations

- **Roboflow API**: Network latency (typically 500-1000ms per call)
- **Polygon Processing**: Real-time (<10ms)
- **Homography Computation**: Real-time (<5ms)
- **Recommendation**: Run calibration only on first frame, cache results

## Limitations

- Requires Roboflow API access (network connection needed)
- Court must be clearly visible in frame
- Works best with static camera (no pan/tilt during video)
- Detection quality depends on lighting and camera angle
- API rate limits may apply (check Roboflow plan)

## Dependencies on Other Modules

This module provides data to:
- **Ball Tracking**: Floor homography for coordinate transformation
- **Player Tracking**: Floor homography for position transformation
- **Wall Hit Detection**: Floor/wall homography for hit position analysis
- **Shot Classification**: Court dimensions and homographies for spatial analysis
- **All Modules**: Wall color detection determines ball preprocessing settings

**Note**: Court calibration is typically run once on the first frame, with results cached for use throughout video processing.

## Author

Youssef Elhagg (yousseframi@aucegypt.edu)

## License

MIT License

This module is part of the SquashCoachingCopilot thesis project at the American University in Cairo.
