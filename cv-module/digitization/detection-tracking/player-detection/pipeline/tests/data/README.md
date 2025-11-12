# Test Data Directory

This directory contains test data for evaluating the player tracking pipeline.

## Required Files

- `_annotations.coco.json` - COCO format annotations for ground truth player positions (already included)
- `clip2.mp4` - Test video file (place your test video here)

## Usage

1. Place your test video file (`clip2.mp4`) in this directory
2. Ensure the COCO annotations match the frames in your test video
3. Run the evaluator from the tests directory:
   ```bash
   cd pipeline/tests
   python evaluator.py
   ```

## Notes

- Test data files are not included in the package distribution
- These files are only used for development and evaluation purposes
