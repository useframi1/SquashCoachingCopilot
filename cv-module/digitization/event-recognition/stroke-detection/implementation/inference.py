import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import pickle

# Import your preprocessing functions
# from your_preprocessing_module import normalize_keypoints_df, add_angle_features, etc.


class StrokePredictor:
    def __init__(self, model_path, yolo_model_path="yolo11n-pose.pt", window_size=16):
        """
        Initialize the stroke predictor

        Args:
            model_path: Path to your trained model (XGBoost .pkl or LSTM .h5)
            yolo_model_path: Path to YOLO pose model
            window_size: Number of frames for prediction window
        """
        self.window_size = window_size
        self.yolo_model = YOLO(yolo_model_path)

        # Load your trained model
        if model_path.endswith(".h5") or model_path.endswith(".keras"):
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
            self.model_type = "lstm"
        elif model_path.endswith(".pkl"):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_type = "xgboost"
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        # COCO keypoint indices
        self.RELEVANT_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
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

        # Buffers for each tracked player (stores ALL keypoints, not just window_size)
        self.player_buffers = {}
        self.last_predictions = {}
        
        # Track prediction cooldown to avoid spam
        self.prediction_cooldown = {}
        self.cooldown_frames = 5  # Don't predict same stroke again for 5 frames

    def extract_keypoints(self, person_keypoints):
        """Extract keypoints into a dictionary"""
        keypoints = {}
        for j, idx in enumerate(self.RELEVANT_IDX):
            kp = person_keypoints[idx].cpu().numpy().tolist()
            keypoints[f"x_{self.RELEVANT_NAMES[j]}"] = kp[0]
            keypoints[f"y_{self.RELEVANT_NAMES[j]}"] = kp[1]
        return keypoints

    def normalize_keypoints(self, keypoints_dict):
        """Normalize keypoints relative to hip center and torso length"""
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
            prediction: Stroke type (forehand/backhand/neither)
            confidence: Prediction confidence
        """
        if len(keypoints_sequence) < self.window_size:
            return None, None

        # SLIDING WINDOW: Take the LAST window_size frames (most recent)
        # This creates an overlapping window that slides with each new frame
        sequence = keypoints_sequence[-self.window_size:]

        # Convert to feature array
        coord_cols = [
            f"{axis}_{name}" for name in self.RELEVANT_NAMES for axis in ["x", "y"]
        ]

        # Create feature matrix
        features = np.array([[frame[col] for col in coord_cols] for frame in sequence])

        if self.model_type == "xgboost":
            # Flatten for XGBoost
            features = features.flatten().reshape(1, -1)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        else:  # LSTM
            # Reshape for LSTM (1, window_size, num_features)
            features = features.reshape(1, self.window_size, -1)
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
        if stroke_type != "neither" and confidence > 0.5:
            self.prediction_cooldown[player_id] = (current_frame, stroke_type)
            return stroke_type, confidence

        return None, None

    def process_frame(self, frame, frame_idx):
        """Process a single frame and return predictions"""
        results = self.yolo_model.track(
            source=frame, persist=True, conf=0.6, verbose=False
        )
        result = results[0]

        keypoints_data = result.keypoints
        ids = (
            result.boxes.id
            if result.boxes is not None and hasattr(result.boxes, "id")
            else None
        )

        predictions = []

        if keypoints_data is not None and ids is not None:
            for i, person_keypoints in enumerate(keypoints_data.data):
                if len(person_keypoints) <= max(self.RELEVANT_IDX):
                    continue

                current_id = int(ids[i].item())

                # Extract and normalize keypoints
                keypoints = self.extract_keypoints(person_keypoints)
                normalized_keypoints = self.normalize_keypoints(keypoints)

                # Initialize buffer for new player (unlimited size for sliding window)
                if current_id not in self.player_buffers:
                    self.player_buffers[current_id] = []
                    self.last_predictions[current_id] = None

                # Add to buffer (sliding window: keep adding frames)
                self.player_buffers[current_id].append(normalized_keypoints)

                # Predict on EVERY frame once we have enough data (sliding window)
                if len(self.player_buffers[current_id]) >= self.window_size:
                    stroke, confidence = self.predict_stroke(
                        current_id, 
                        self.player_buffers[current_id],
                        frame_idx
                    )

                    # Report new predictions
                    if stroke is not None:
                        predictions.append(
                            {
                                "player_id": current_id,
                                "stroke": stroke,
                                "confidence": confidence,
                                "frame": frame_idx,
                            }
                        )
                        self.last_predictions[current_id] = stroke

        return predictions, result.plot()


def run_inference(video_path, model_path, output_path=None):
    """
    Run inference on a video

    Args:
        video_path: Path to input video
        model_path: Path to trained model
        output_path: Optional path to save output video
    """
    predictor = StrokePredictor(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer for output
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    print("Starting inference with SLIDING WINDOW...")
    print("Press 'q' to quit")
    print("-" * 60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Process frame
        predictions, annotated_frame = predictor.process_frame(frame, frame_idx)

        # Print predictions
        for pred in predictions:
            print(
                f"Frame {pred['frame']:5d} | Player {pred['player_id']} did {pred['stroke'].upper()} (confidence: {pred['confidence']:.2f})"
            )

        # Add frame number overlay
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        # Add text overlay for predictions
        y_offset = 70
        for player_id, last_stroke in predictor.last_predictions.items():
            if last_stroke and last_stroke != "neither":
                text = f"Player {player_id}: {last_stroke.upper()}"
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                y_offset += 30

        # Write frame
        if writer:
            writer.write(annotated_frame)

        # Display
        cv2.imshow("Stroke Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("-" * 60)
    print("Inference complete!")


if __name__ == "__main__":
    # Example usage
    video_path = "/home/g03-s2025/Desktop/SquashCoachingCopilot/cv-module/digitization/event-recognition/stroke-detection/implementation/Videos/video-2.mp4"
    model_path = "/home/g03-s2025/Desktop/SquashCoachingCopilot/cv-module/digitization/event-recognition/stroke-detection/implementation/lstm_model.h5"  # or .h5 for LSTM
    output_path = "/home/g03-s2025/Desktop/SquashCoachingCopilot/cv-module/digitization/event-recognition/stroke-detection/implementation/Videos/output_video.mp4"  # Optional

    run_inference(video_path, model_path, output_path)