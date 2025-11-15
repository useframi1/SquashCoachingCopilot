"""
Common utility functions for the SquashCopilot package.

This module provides shared utility functions used across all modules,
including configuration loading and package directory management.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def load_config(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    This function supports two modes:
    1. Explicit path: Provide config_path to load from a specific file
    2. Default config: Provide config_name to load from squashcopilot/configs/

    Args:
        config_path: Explicit path to a config YAML file. If provided, config_name is ignored.
        config_name: Name of the config file (without .yaml extension) in squashcopilot/configs/.
                    For example: 'ball_tracking', 'player_tracking', etc.

    Returns:
        Dictionary containing the configuration

    Raises:
        ValueError: If neither config_path nor config_name is provided
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed

    Examples:
        >>> # Load default config
        >>> config = load_config(config_name='ball_tracking')

        >>> # Load from explicit path
        >>> config = load_config(config_path='/path/to/custom_config.yaml')
    """
    if config_path is None and config_name is None:
        raise ValueError("Either config_path or config_name must be provided")

    if config_path is not None:
        # Load from explicit path
        config_file = Path(config_path)
    else:
        # Load from default configs directory
        # Get the package root (squashcopilot/)
        package_root = Path(__file__).parent.parent
        config_file = package_root / "configs" / f"{config_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_package_dir(module_path: str) -> str:
    """
    Get the absolute path to the directory containing a module file.

    This is useful for constructing paths relative to a module's location,
    such as model weights or other resources.

    Args:
        module_path: The __file__ attribute from the calling module

    Returns:
        Absolute path to the directory containing the module file

    Example:
        >>> # In a module file
        >>> package_dir = get_package_dir(__file__)
        >>> model_path = os.path.join(package_dir, 'models', 'weights', 'model.pt')
    """
    return str(Path(module_path).parent.absolute())
