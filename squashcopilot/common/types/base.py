"""
Base types for the squash coaching copilot system.

This module defines fundamental types used across all modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class Frame:
    """
    Represents a video frame with metadata.

    Attributes:
        image: BGR image as numpy array
        frame_number: Frame index in video (0-based)
        timestamp: Timestamp in seconds
        original_shape: Original image shape (height, width, channels)
    """
    image: np.ndarray
    frame_number: int
    timestamp: float
    original_shape: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))

    def __post_init__(self):
        """Set original_shape from image if not provided."""
        if self.original_shape == (0, 0, 0) and self.image is not None:
            self.original_shape = self.image.shape

    @property
    def height(self) -> int:
        """Get frame height."""
        return self.original_shape[0]

    @property
    def width(self) -> int:
        """Get frame width."""
        return self.original_shape[1]

    @property
    def channels(self) -> int:
        """Get number of color channels."""
        return self.original_shape[2] if len(self.original_shape) > 2 else 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding image data)."""
        return {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'original_shape': self.original_shape
        }


@dataclass
class Config:
    """
    Configuration container for module parameters.

    Attributes:
        params: Dictionary of configuration parameters
    """
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'Config':
        """
        Create Config from dictionary.

        Args:
            d: Dictionary of parameters (can be None)

        Returns:
            Config instance
        """
        if d is None:
            return cls()
        return cls(params=d)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Parameter key (supports nested keys with dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Support nested keys like 'postprocessing.window'
        keys = key.split('.')
        value = self.params

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Parameter key (supports nested keys with dot notation)
            value: Value to set
        """
        keys = key.split('.')
        target = self.params

        # Navigate to parent
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set value
        target[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.params.copy()

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
