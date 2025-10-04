import numpy as np
from tensorflow import keras


class StrokeDetector:
    """
    LSTM-based Stroke Detection Module for Tennis/Squash
    Processes keypoints from an external source and predicts stroke types using sliding window.
    """

    def __init__(self, model_path, window_size=16, confidence_threshold=0.5, cooldown_frames=5):
        """
        Initialize the stroke detector

        Args:
            model_path: Path to your trained LSTM model (.h5 or .keras)
            window_size: Number of frames for prediction window (default: 16)
            confidence_threshold: Minimum confidence to report prediction (default: 0.5)
            cooldown_frames: Number of frames to wait before reporting same stroke again (default: 5)
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames

        # Load LSTM model
        if model_path.endswith(".h5") or model_path.endswith(".keras"):
            self.model = keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format. Expected .h5 or .keras, got: {model_path}")

        # COCO keypoint names (indices 5-16 in COCO format)
        self.RELEVANT_NAMES = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        # Label mapping
        self.label_map = {0: "forehand", 1: "backhand", 2: "neither"}

        # Buffers for each tracked player
        self.player_buffers = {}
        self.last_predictions = {}
        self.prediction_cooldown = {}

    def extract_relevant_keypoints(self, person_keypoints):
        """Extract keypoints into a dictionary"""
        keypoints = {}
        for j, idx in enumerate(self.RELEVANT_IDX):
            kp = person_keypoints[idx].cpu().numpy().tolist()
            keypoints[f"x_{self.RELEVANT_NAMES[j]}"] = kp[0]
            keypoints[f"y_{self.RELEVANT_NAMES[j]}"] = kp[1]
        return keypoints


    def normalize_keypoints(self, keypoints_dict):
        """
        Normalize keypoints relative to hip center and torso length
        
        Args:
            keypoints_dict: Dictionary with x_ and y_ coordinates for relevant keypoints
            
        Returns:
            Dictionary with normalized keypoints
        """
        # Calculate hip center
        hip_center_x = (
            keypoints_dict["x_left_hip"] + keypoints_dict["x_right_hip"]
        ) / 2
        hip_center_y = (
            keypoints_dict["y_left_hip"] + keypoints_dict["y_right_hip"]
        ) / 2

        # Calculate shoulder center
        shoulder_center_x = (
            keypoints_dict["x_left_shoulder"] + keypoints_dict["x_right_shoulder"]
        ) / 2
        shoulder_center_y = (
            keypoints_dict["y_left_shoulder"] + keypoints_dict["y_right_shoulder"]
        ) / 2

        # Calculate torso length
        torso_length = np.sqrt(
            (shoulder_center_x - hip_center_x) ** 2
            + (shoulder_center_y - hip_center_y) ** 2
        )

        # Avoid division by zero
        if torso_length < 1e-6:
            torso_length = 1.0

        # Normalize all keypoints
        normalized = {}
        for name in self.RELEVANT_NAMES:
            normalized[f"x_{name}"] = (
                keypoints_dict[f"x_{name}"] - hip_center_x
            ) / torso_length
            normalized[f"y_{name}"] = (
                keypoints_dict[f"y_{name}"] - hip_center_y
            ) / torso_length

        return normalized

    def predict_stroke(self, player_id, keypoints_sequence, current_frame):
        """
        Predict stroke type from a sequence of keypoints using sliding window

        Args:
            player_id: ID of the player
            keypoints_sequence: List of ALL normalized keypoint dictionaries for this player
            current_frame: Current frame number

        Returns:
            tuple: (stroke_type, confidence) or (None, None) if no prediction
        """
        if len(keypoints_sequence) < self.window_size:
            return None, None

        # SLIDING WINDOW: Take the LAST window_size frames (most recent)
        sequence = keypoints_sequence[-self.window_size:]

        # Convert to feature array
        coord_cols = [
            f"{axis}_{name}" for name in self.RELEVANT_NAMES for axis in ["x", "y"]
        ]

        # Create feature matrix (window_size, num_features)
        features = np.array([[frame[col] for col in coord_cols] for frame in sequence])

        # Reshape for LSTM (1, window_size, num_features)
        features = features.reshape(1, self.window_size, -1)
        
        # Get prediction
        probabilities = self.model.predict(features, verbose=0)[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]

        stroke_type = self.label_map[prediction]

        # Check cooldown to avoid repetitive predictions
        if player_id in self.prediction_cooldown:
            last_frame, last_stroke = self.prediction_cooldown[player_id]
            if current_frame - last_frame < self.cooldown_frames and stroke_type == last_stroke:
                return None, None

        # Only report if not "neither" and confidence > threshold
        if stroke_type != "neither" and confidence > self.confidence_threshold:
            self.prediction_cooldown[player_id] = (current_frame, stroke_type)
            return stroke_type, confidence

        return None, None

    def process_frame(self, player_keypoints, frame_idx):
        """
        Process keypoints for multiple players and return predictions
        
        Args:
            player_keypoints: Dictionary mapping player_id to their keypoints
                             Format: {player_id: keypoints_data, ...}
                             where keypoints_data can be:
                               - Dictionary: {'x_left_shoulder': val, 'y_left_shoulder': val, ...}
                               - Array/List: COCO format keypoints (17 keypoints)
            frame_idx: Current frame number
            
        Returns:
            List of prediction dictionaries with keys:
                - player_id: ID of the player
                - stroke: Predicted stroke type ('forehand', 'backhand')
                - confidence: Prediction confidence (0-1)
                - frame: Frame number
        """
        predictions = []

        for player_id, keypoints_data in player_keypoints.items():
            try:
                # Extract relevant keypoints
                keypoints = self.extract_relevant_keypoints(keypoints_data)
                
                # Normalize keypoints
                normalized_keypoints = self.normalize_keypoints(keypoints)

                # Initialize buffer for new player
                if player_id not in self.player_buffers:
                    self.player_buffers[player_id] = []
                    self.last_predictions[player_id] = None

                # Add to buffer (sliding window: keep adding frames)
                self.player_buffers[player_id].append(normalized_keypoints)

                # Predict on EVERY frame once we have enough data (sliding window)
                if len(self.player_buffers[player_id]) >= self.window_size:
                    stroke, confidence = self.predict_stroke(
                        player_id,
                        self.player_buffers[player_id],
                        frame_idx
                    )

                    # Report new predictions
                    if stroke is not None:
                        predictions.append({
                            "player_id": player_id,
                            "stroke": stroke,
                            "confidence": confidence,
                            "frame": frame_idx,
                        })
                        self.last_predictions[player_id] = stroke

            except Exception as e:
                print(f"Warning: Error processing player {player_id} at frame {frame_idx}: {e}")
                continue

        return predictions


# Example usage
# if __name__ == "__main__":
#     # Initialize the stroke detector
#     model_path = "lstm_model.h5"  # or .keras
#     detector = StrokeDetector(
#         model_path=model_path,
#         window_size=16,
#         confidence_threshold=0.5,
#         cooldown_frames=5
#     )

#     # Simulate processing frames
#     for frame_idx in range(1, 100):
#         # Example: Get keypoints from your tracking module
#         # Format 1: Dictionary format
#         player_keypoints = {
#             1: {  # Player ID 1
#                 'x_left_shoulder': 100, 'y_left_shoulder': 200,
#                 'x_right_shoulder': 150, 'y_right_shoulder': 200,
#                 'x_left_elbow': 90, 'y_left_elbow': 250,
#                 'x_right_elbow': 160, 'y_right_elbow': 250,
#                 'x_left_wrist': 80, 'y_left_wrist': 300,
#                 'x_right_wrist': 170, 'y_right_wrist': 300,
#                 'x_left_hip': 110, 'y_left_hip': 350,
#                 'x_right_hip': 140, 'y_right_hip': 350,
#                 'x_left_knee': 105, 'y_left_knee': 450,
#                 'x_right_knee': 145, 'y_right_knee': 450,
#                 'x_left_ankle': 100, 'y_left_ankle': 550,
#                 'x_right_ankle': 150, 'y_right_ankle': 550,
#             },
#             # Format 2: Array format (COCO)
#             2: np.array([  # Player ID 2 - 17 keypoints in COCO format
#                 [120, 50],   # nose
#                 [115, 45],   # left_eye
#                 [125, 45],   # right_eye
#                 [110, 50],   # left_ear
#                 [130, 50],   # right_ear
#                 [100, 100],  # left_shoulder (index 5)
#                 [140, 100],  # right_shoulder
#                 [90, 150],   # left_elbow
#                 [150, 150],  # right_elbow
#                 [85, 200],   # left_wrist
#                 [155, 200],  # right_wrist
#                 [105, 250],  # left_hip
#                 [135, 250],  # right_hip
#                 [100, 350],  # left_knee
#                 [140, 350],  # right_knee
#                 [95, 450],   # left_ankle
#                 [145, 450],  # right_ankle
#             ])
#         }

#         # Process keypoints and get predictions
#         predictions = detector.process_frame(player_keypoints, frame_idx)

#         # Print predictions
#         for pred in predictions:
#             print(
#                 f"Frame {pred['frame']:5d} | Player {pred['player_id']} did "
#                 f"{pred['stroke'].upper()} (confidence: {pred['confidence']:.2f})"
#             )

#     # Get last predictions for all players
#     last_predictions = detector.get_last_predictions()
#     print(f"\nFinal predictions: {last_predictions}")