# Court Calibration Tests

This directory contains evaluation tools for testing the court detection and calibration pipeline.

## Structure

```
tests/
├── config.json              # Test configuration
├── evaluator.py            # Evaluation script
├── data/                   # Test data directory
│   ├── README.md
│   └── test_frame.jpg      # Test image (add your own)
└── outputs/                # Evaluation outputs
    └── calibration_output.jpg
```

## Setup

1. Place a test image in the `data/` directory:
   ```bash
   cp /path/to/your/frame.jpg data/test_frame.jpg
   ```

2. Adjust test points in `config.json` if needed

## Running the Evaluation

```bash
cd tests
python evaluator.py
```

## What the Evaluator Does

1. **Court Detection**: Detects all court elements (tin, service boxes, front wall)
2. **Homography Computation**: Computes two homography matrices:
   - **Floor homography**: Maps floor plane to real-world coordinates (uses 10 points)
   - **Wall homography**: Maps wall plane to real-world coordinates (uses 4 points)

3. **Wall Color Detection**: Analyzes the front wall to determine:
   - Whether the wall is white or dark/colored
   - Recommended ball color (black for white walls, yellow for dark walls)
   - Color statistics (brightness and saturation)

4. **Inverse Transform Verification**: Tests the homographies by:
   - Taking known real-world coordinates (in meters)
   - Applying inverse transform to get pixel coordinates
   - Visualizing the results on the court image

5. **Visualization Output**: Creates an annotated image showing:
   - Detected keypoints (magenta polygons)
   - Wall color info overlay (top-right corner)
   - Floor test points (green/red markers)
   - Wall test points (blue/yellow markers)

## Configuration

Edit `config.json` to customize:

### Test Points

Define real-world coordinates to verify the homographies:

```json
"test_points": {
    "floor": [
        {
            "real": [3.2, 6.24],
            "description": "Center of court at service line",
            "color": [0, 255, 0]
        }
    ],
    "wall": [
        {
            "real": [3.2, 1.14],
            "description": "Center of front wall middle section",
            "color": [0, 0, 255]
        }
    ]
}
```

### Coordinate System

**Floor coordinates** (x, y in meters):
- Origin (0, 0) at bottom-left corner of court
- x-axis: left to right (0 to 6.4m)
- y-axis: front wall to back wall (0 to ~9.75m)

**Wall coordinates** (x, z in meters):
- Origin (0, 0) at bottom-left of front wall
- x-axis: left to right (0 to 6.4m)
- z-axis: floor to ceiling (0 to ~5m)

## Interpreting Results

The evaluator prints:
- Homography matrices for floor and wall
- Detected keypoints for each court element
- Transformed pixel coordinates for each test point

Check the output image (`outputs/calibration_output.jpg`):
- **Magenta polygons**: Detected court elements (verify detection accuracy)
- **Colored markers**: Test points transformed from real-world coordinates
  - Markers should align with the expected court locations
  - If misaligned, the homography may need more/better keypoints

## Example Test Points

Good test points to verify calibration:

**Floor**:
- Court center: [3.2, 6.24]
- Service box corners: [0, 5.44], [1.6, 7.04], [4.8, 5.44], [6.4, 7.04]
- Back wall center: [3.2, 9.75]

**Wall**:
- Tin top center: [3.2, 0.48]
- Service line: [3.2, 1.78]
- Court line (lower): [3.2, 2.13]

## Troubleshooting

**Court elements not detected**:
- Check image quality and lighting
- Verify court markings are clearly visible
- Adjust confidence threshold in court_calibration config

**Test points misaligned**:
- Verify real-world coordinates are correct
- Check that court dimensions match standard squash court
- Ensure camera angle is stable and not distorted

**Homography computation fails**:
- Ensure all required court elements are detected
- Check that keypoints form valid quadrilaterals
- Verify epsilon_factor in calibration config
