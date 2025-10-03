import json
from importlib import resources
import os


def load_config(config_path=None):
    if config_path is None:
        # Read from package resources
        try:
            # Python 3.9+
            config_file = resources.files("player_tracking_pipeline").joinpath(
                "config.json"
            )
            with config_file.open("r") as f:
                config = json.load(f)
        except AttributeError:
            # Python 3.7-3.8 fallback
            with resources.open_text("player_tracking_pipeline", "config.json") as f:
                config = json.load(f)
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
    return config


def get_package_dir():
    """Get the absolute path to the player_tracking_pipeline package root"""
    # Go up one level to get to player_tracking_pipeline/
    return os.path.dirname(os.path.abspath(__file__))
