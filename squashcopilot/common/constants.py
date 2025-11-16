"""
Common constants used throughout the squash coaching copilot system.
"""

# ============================================================================
# COCO Keypoint Constants
# ============================================================================

# COCO keypoint indices for body parts (excluding face)
# Indices 5-16: shoulders, elbows, wrists, hips, knees, ankles
BODY_KEYPOINT_INDICES = list(range(5, 17))

# Names for body keypoints (indices 5-16)
KEYPOINT_NAMES = [
    'left_shoulder',   # index 5
    'right_shoulder',  # index 6
    'left_elbow',      # index 7
    'right_elbow',     # index 8
    'left_wrist',      # index 9
    'right_wrist',     # index 10
    'left_hip',        # index 11
    'right_hip',       # index 12
    'left_knee',       # index 13
    'right_knee',      # index 14
    'left_ankle',      # index 15
    'right_ankle',     # index 16
]

# COCO skeleton connections for visualization
# Each tuple represents (start_keypoint_index, end_keypoint_index)
SKELETON_CONNECTIONS = [
    (5, 6),    # shoulders
    (5, 7),    # left shoulder to left elbow
    (7, 9),    # left elbow to left wrist
    (6, 8),    # right shoulder to right elbow
    (8, 10),   # right elbow to right wrist
    (5, 11),   # left shoulder to left hip
    (6, 12),   # right shoulder to right hip
    (11, 12),  # hips
    (11, 13),  # left hip to left knee
    (13, 15),  # left knee to left ankle
    (12, 14),  # right hip to right knee
    (14, 16),  # right knee to right ankle
]

# Full COCO keypoint names (all 17 keypoints)
# This includes face keypoints (0-4) which are typically not used for squash
COCO_KEYPOINT_NAMES_FULL = [
    'nose',            # index 0
    'left_eye',        # index 1
    'right_eye',       # index 2
    'left_ear',        # index 3
    'right_ear',       # index 4
    'left_shoulder',   # index 5
    'right_shoulder',  # index 6
    'left_elbow',      # index 7
    'right_elbow',     # index 8
    'left_wrist',      # index 9
    'right_wrist',     # index 10
    'left_hip',        # index 11
    'right_hip',       # index 12
    'left_knee',       # index 13
    'right_knee',      # index 14
    'left_ankle',      # index 15
    'right_ankle',     # index 16
]
