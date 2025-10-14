import json
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_and_combine_data(directory: str) -> pd.DataFrame:
    """
    Reads all CSV files from the given directory and appends them into a single DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame containing all CSV data.
    """
    all_dfs = []

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_csv(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty DataFrame if no CSVs found


def postprocess_predictions(
    predictions: pd.Series,
    min_duration: dict = None,
) -> pd.Series:
    """
    Remove fluctuations while enforcing state transition rules.

    A state change is only accepted if:
    1. It follows valid transition rules (FSM)
    2. The new state persists for at least min_duration[state] frames

    This removes noise and brief fluctuations in predictions while keeping
    legitimate state transitions.

    Args:
        predictions: Series of predicted states ("start", "active", "end")
        min_duration: Dictionary mapping state names to minimum consecutive frames required.
                     Default: {"start": 5, "active": 15, "end": 5}

    Returns:
        Series of postprocessed predictions with fluctuations removed

    Example:
        Input:  ["active", "active", "end", "active", "active"]
        If "end" min_duration=3, and it only appears once, it's a fluctuation.
        Output: ["active", "active", "active", "active", "active"]
    """
    # Default minimum durations
    if min_duration is None:
        min_duration = {
            "start": 5,
            "active": 15,
            "end": 5,
        }

    # Valid state transitions (Finite State Machine)
    valid_transitions = {
        "start": ["start", "active"],
        "active": ["active", "end"],
        "end": ["end", "start"],
    }

    postprocessed = predictions.copy()

    # Always start with "start" state
    current_committed_state = "start"
    postprocessed.iloc[0] = current_committed_state

    i = 1
    while i < len(predictions):
        candidate_state = predictions.iloc[i]

        # Check if this is a valid transition
        if candidate_state not in valid_transitions[current_committed_state]:
            # Invalid transition - replace with current committed state
            postprocessed.iloc[i] = current_committed_state
            i += 1
            continue

        # If same state as current, just continue
        if candidate_state == current_committed_state:
            postprocessed.iloc[i] = current_committed_state
            i += 1
            continue

        # Valid new state - check if it persists long enough
        # Count consecutive occurrences of this candidate state
        count = 1
        j = i + 1
        while j < len(predictions) and predictions.iloc[j] == candidate_state:
            count += 1
            j += 1

        # Check if candidate state meets minimum duration requirement
        if count >= min_duration.get(candidate_state, 3):
            # Real transition - commit it
            for k in range(i, j):
                postprocessed.iloc[k] = candidate_state
            current_committed_state = candidate_state
            i = j
        else:
            # Fluctuation - reject it and replace with current committed state
            for k in range(i, j):
                postprocessed.iloc[k] = current_committed_state
            i = j

    return postprocessed


def apply_tolerance_to_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    tolerance_frames: int = 3,
) -> pd.Series:
    """
    Apply temporal tolerance to predictions for frame-level evaluation.

    For each frame where prediction doesn't match ground truth, check if
    the ground truth value appears within ±tolerance_frames window. If yes,
    consider it correct by adjusting the prediction.

    Args:
        y_true: Ground truth state sequence
        y_pred: Predicted state sequence
        tolerance_frames: Number of frames to look ahead/behind for tolerance

    Returns:
        Adjusted predictions where mismatches within tolerance are corrected
    """
    y_pred_tolerant = y_pred.copy()

    for i in range(len(y_true)):
        if y_pred.iloc[i] == y_true.iloc[i]:
            # Already correct, no adjustment needed
            continue

        # Check within tolerance window
        window_start = max(0, i - tolerance_frames)
        window_end = min(len(y_true) - 1, i + tolerance_frames)

        # Look for the predicted state in the ground truth window
        predicted_state = y_pred.iloc[i]
        for j in range(window_start, window_end + 1):
            if y_true.iloc[j] == predicted_state:
                # Found the predicted state within tolerance window
                # This means the prediction is "close enough", so mark as correct
                y_pred_tolerant.iloc[i] = y_true.iloc[i]
                break

    return y_pred_tolerant
