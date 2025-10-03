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
            config_file = resources.files("court_detection_pipeline").joinpath(
                "config.json"
            )
            with config_file.open("r") as f:
                config = json.load(f)
        except AttributeError:
            # Python 3.7-3.8 fallback
            with resources.open_text("court_detection_pipeline", "config.json") as f:
                config = json.load(f)
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
    return config


def get_package_dir():
    """Get the absolute path to the court_detection_pipeline package root"""
    return os.path.dirname(os.path.abspath(__file__))
