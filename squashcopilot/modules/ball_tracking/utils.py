"""Ball tracking specific utility functions."""

import numpy as np
import cv2


def postprocess(feature_map, scale=2):
    """
    Postprocess the TrackNet output heatmap to extract ball position.

    Uses Hough Circle detection to find the ball center from the model's
    feature map output.

    Args:
        feature_map: Model output feature map (normalized 0-1)
        scale: Scale factor to convert from model resolution to original resolution

    Returns:
        Tuple of (x, y) coordinates or (None, None) if no ball detected
    """
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0] * scale
            y = circles[0][0][1] * scale
    return x, y
