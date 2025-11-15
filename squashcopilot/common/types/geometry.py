"""
Geometric types for the squash coaching copilot system.

This module defines geometric primitives used for spatial calculations
and coordinate transformations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Point2D:
    """
    Represents a 2D point in pixel or real-world coordinates.

    Attributes:
        x: X coordinate
        y: Y coordinate
    """
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (x, y)."""
        return (self.x, self.y)

    def to_list(self) -> List[float]:
        """Convert to list [x, y]."""
        return [self.x, self.y]

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y], dtype=np.float32)

    def distance_to(self, other: 'Point2D') -> float:
        """
        Calculate Euclidean distance to another point.

        Args:
            other: Another Point2D

        Returns:
            Distance in same units as coordinates
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {'x': self.x, 'y': self.y}

    @classmethod
    def from_tuple(cls, t: Optional[Tuple[float, float]]) -> Optional['Point2D']:
        """
        Create Point2D from tuple.

        Args:
            t: Tuple (x, y) or None

        Returns:
            Point2D instance or None
        """
        if t is None:
            return None
        return cls(x=t[0], y=t[1])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Point2D':
        """
        Create Point2D from numpy array.

        Args:
            arr: Numpy array of shape (2,)

        Returns:
            Point2D instance
        """
        return cls(x=float(arr[0]), y=float(arr[1]))

    def __repr__(self) -> str:
        return f"Point2D(x={self.x:.2f}, y={self.y:.2f})"


@dataclass
class BoundingBox:
    """
    Represents a rectangular bounding box.

    Attributes:
        x1: Left edge
        y1: Top edge
        x2: Right edge
        y2: Bottom edge
    """
    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> List[float]:
        """Convert to list [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        """Get center point of bounding box."""
        return Point2D(
            x=(self.x1 + self.x2) / 2,
            y=(self.y1 + self.y2) / 2
        )

    @property
    def bottom_center(self) -> Point2D:
        """Get bottom-center point (useful for player position)."""
        return Point2D(
            x=(self.x1 + self.x2) / 2,
            y=self.y2
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'width': self.width,
            'height': self.height
        }

    @classmethod
    def from_list(cls, lst: List[float]) -> 'BoundingBox':
        """
        Create BoundingBox from list.

        Args:
            lst: List [x1, y1, x2, y2]

        Returns:
            BoundingBox instance
        """
        return cls(x1=lst[0], y1=lst[1], x2=lst[2], y2=lst[3])

    def __repr__(self) -> str:
        return f"BoundingBox(x1={self.x1:.1f}, y1={self.y1:.1f}, x2={self.x2:.1f}, y2={self.y2:.1f})"


@dataclass
class Homography:
    """
    Represents a homography transformation matrix.

    Attributes:
        matrix: 3x3 transformation matrix
        source_plane: Name of the source plane (e.g., 'floor', 'wall')
        target_plane: Name of the target plane (default: 'world')
    """
    matrix: np.ndarray
    source_plane: str
    target_plane: str = 'world'

    def __post_init__(self):
        """Validate matrix shape."""
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {self.matrix.shape}")

    def transform_point(self, point: Point2D) -> Point2D:
        """
        Transform a point using this homography.

        Args:
            point: Point in source plane

        Returns:
            Transformed point in target plane
        """
        # Convert to homogeneous coordinates
        pt_homo = np.array([point.x, point.y, 1.0])

        # Apply transformation
        transformed = self.matrix @ pt_homo

        # Convert back from homogeneous coordinates
        if transformed[2] != 0:
            x = transformed[0] / transformed[2]
            y = transformed[1] / transformed[2]
        else:
            x, y = transformed[0], transformed[1]

        return Point2D(x=x, y=y)

    def transform_points(self, points: List[Point2D]) -> List[Point2D]:
        """
        Transform multiple points.

        Args:
            points: List of points in source plane

        Returns:
            List of transformed points in target plane
        """
        return [self.transform_point(p) for p in points]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'matrix': self.matrix.tolist(),
            'source_plane': self.source_plane,
            'target_plane': self.target_plane
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Homography':
        """
        Create Homography from dictionary.

        Args:
            d: Dictionary with matrix, source_plane, target_plane

        Returns:
            Homography instance
        """
        return cls(
            matrix=np.array(d['matrix'], dtype=np.float32),
            source_plane=d['source_plane'],
            target_plane=d.get('target_plane', 'world')
        )

    def __repr__(self) -> str:
        return f"Homography(source={self.source_plane}, target={self.target_plane})"


@dataclass
class Keypoints:
    """
    Represents a collection of named keypoints with optional confidences.

    Attributes:
        points: Dictionary mapping keypoint names to Point2D
        confidences: Optional dictionary mapping keypoint names to confidence scores
    """
    points: Dict[str, Point2D]
    confidences: Optional[Dict[str, float]] = None

    def get_point(self, name: str) -> Optional[Point2D]:
        """
        Get a specific keypoint by name.

        Args:
            name: Keypoint name

        Returns:
            Point2D or None if not found
        """
        return self.points.get(name)

    def get_confidence(self, name: str) -> Optional[float]:
        """
        Get confidence for a specific keypoint.

        Args:
            name: Keypoint name

        Returns:
            Confidence score or None
        """
        if self.confidences is None:
            return None
        return self.confidences.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'points': {name: point.to_dict() for name, point in self.points.items()}
        }
        if self.confidences is not None:
            result['confidences'] = self.confidences
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Keypoints':
        """
        Create Keypoints from dictionary.

        Args:
            d: Dictionary with points and optional confidences

        Returns:
            Keypoints instance
        """
        points = {
            name: Point2D(**point_dict)
            for name, point_dict in d['points'].items()
        }
        confidences = d.get('confidences')
        return cls(points=points, confidences=confidences)

    @classmethod
    def from_array(cls, arr: np.ndarray, names: Optional[List[str]] = None) -> 'Keypoints':
        """
        Create Keypoints from numpy array.

        Args:
            arr: Array of shape (N, 2) or (N, 3) where N is number of keypoints
                  If shape is (N, 3), third column is confidence
            names: Optional list of keypoint names (defaults to "point_0", "point_1", ...)

        Returns:
            Keypoints instance
        """
        n_points = arr.shape[0]

        # Generate names if not provided
        if names is None:
            names = [f"point_{i}" for i in range(n_points)]

        # Extract points
        points = {
            names[i]: Point2D(x=arr[i, 0], y=arr[i, 1])
            for i in range(n_points)
        }

        # Extract confidences if available
        confidences = None
        if arr.shape[1] >= 3:
            confidences = {names[i]: float(arr[i, 2]) for i in range(n_points)}

        return cls(points=points, confidences=confidences)

    def to_array(self, include_confidence: bool = False) -> np.ndarray:
        """
        Convert to numpy array.

        Args:
            include_confidence: If True, include confidence as third column

        Returns:
            Array of shape (N, 2) or (N, 3)
        """
        names = list(self.points.keys())
        n_points = len(names)

        if include_confidence and self.confidences is not None:
            arr = np.zeros((n_points, 3), dtype=np.float32)
            for i, name in enumerate(names):
                point = self.points[name]
                arr[i, 0] = point.x
                arr[i, 1] = point.y
                arr[i, 2] = self.confidences.get(name, 0.0)
        else:
            arr = np.zeros((n_points, 2), dtype=np.float32)
            for i, name in enumerate(names):
                point = self.points[name]
                arr[i, 0] = point.x
                arr[i, 1] = point.y

        return arr

    def __len__(self) -> int:
        """Get number of keypoints."""
        return len(self.points)

    def __repr__(self) -> str:
        return f"Keypoints(n_points={len(self.points)})"
