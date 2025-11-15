"""Stroke detection specific utility functions for keypoint processing."""

import numpy as np
import torch
from typing import Dict, List, Union


def extract_keypoints(
    keypoints_data: Union[torch.Tensor, np.ndarray, Dict],
    relevant_indices: List[int],
    relevant_names: List[str],
) -> Dict[str, float]:
    """
    Extract relevant keypoints and convert to standardized dictionary format.

    Args:
        keypoints_data: Keypoints in COCO format (tensor/array) or dict format
        relevant_indices: List of COCO keypoint indices to extract
        relevant_names: List of names corresponding to the indices

    Returns:
        Dictionary with format: {'x_left_shoulder': val, 'y_left_shoulder': val, ...}
    """
    # If already in dict format, return as-is
    if isinstance(keypoints_data, dict):
        return keypoints_data

    # Convert tensor/array to dict format
    keypoints = {}
    for j, idx in enumerate(relevant_indices):
        if isinstance(keypoints_data, torch.Tensor):
            kp = keypoints_data[idx].cpu().numpy().tolist()
        else:
            kp = (
                keypoints_data[idx].tolist()
                if hasattr(keypoints_data[idx], "tolist")
                else keypoints_data[idx]
            )

        keypoints[f"x_{relevant_names[j]}"] = kp[0]
        keypoints[f"y_{relevant_names[j]}"] = kp[1]

    return keypoints


def normalize_keypoints(
    keypoints: Dict[str, float],
    relevant_names: List[str],
    min_torso_length: float = 1e-6,
) -> Dict[str, float]:
    """
    Normalize keypoints relative to hip center and torso length.

    This normalization makes the pose representation invariant to player position
    and size, improving model generalization.

    Args:
        keypoints: Dictionary with x_ and y_ coordinates
        relevant_names: List of keypoint names
        min_torso_length: Minimum torso length to avoid division by zero

    Returns:
        Dictionary with normalized keypoints
    """
    # Calculate hip center
    hip_center_x = (keypoints["x_left_hip"] + keypoints["x_right_hip"]) / 2
    hip_center_y = (keypoints["y_left_hip"] + keypoints["y_right_hip"]) / 2

    # Calculate shoulder center
    shoulder_center_x = (
        keypoints["x_left_shoulder"] + keypoints["x_right_shoulder"]
    ) / 2
    shoulder_center_y = (
        keypoints["y_left_shoulder"] + keypoints["y_right_shoulder"]
    ) / 2

    # Calculate torso length
    torso_length = np.sqrt(
        (shoulder_center_x - hip_center_x) ** 2
        + (shoulder_center_y - hip_center_y) ** 2
    )

    # Avoid division by zero
    if torso_length < min_torso_length:
        torso_length = 1.0

    # Normalize all keypoints
    normalized = {}
    for name in relevant_names:
        normalized[f"x_{name}"] = (keypoints[f"x_{name}"] - hip_center_x) / torso_length
        normalized[f"y_{name}"] = (keypoints[f"y_{name}"] - hip_center_y) / torso_length

    return normalized


def prepare_features(
    keypoints_sequence: List[Dict[str, float]],
    relevant_names: List[str],
    window_size: int,
) -> np.ndarray:
    """
    Convert keypoints sequence to feature array for LSTM model input.

    Args:
        keypoints_sequence: List of normalized keypoint dictionaries
        relevant_names: List of keypoint names
        window_size: Size of the prediction window

    Returns:
        numpy array of shape (1, window_size, num_features) ready for LSTM
    """
    # Define feature column order
    coord_cols = [f"{axis}_{name}" for name in relevant_names for axis in ["x", "y"]]

    # Create feature matrix (window_size, num_features)
    features = np.array(
        [[frame[col] for col in coord_cols] for frame in keypoints_sequence]
    )

    # Reshape for LSTM (batch_size=1, window_size, num_features)
    return features.reshape(1, window_size, -1)
