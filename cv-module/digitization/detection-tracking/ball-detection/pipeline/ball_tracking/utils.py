import numpy as np
import cv2
import json
from importlib import resources
import os


def load_config(config_path=None):
    if config_path is None:
        # Read from package resources
        try:
            # Python 3.9+
            config_file = resources.files("ball_tracking").joinpath("config.json")
            with config_file.open("r") as f:
                config = json.load(f)
        except AttributeError:
            # Python 3.7-3.8 fallback
            with resources.open_text("ball_tracking", "config.json") as f:
                config = json.load(f)
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
    return config


def get_package_dir():
    """Get the absolute path to the ball_detection_pipeline package root"""
    return os.path.dirname(os.path.abspath(__file__))


def postprocess(feature_map, scale=2):
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
