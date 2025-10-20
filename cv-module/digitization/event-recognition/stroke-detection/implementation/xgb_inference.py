"""
Real-time XGBoost stroke detection inference with buffered lagging features.
Adapted to match the existing LSTM inference structure.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import joblib
import pandas as pd


class XGBoostStrokePredictor:
    def __init__(self, model_path, yolo_model_path="yolo11n-pose.pt"):
        """
        Initialize the XGBoost stroke predictor

        Args:
            model_path: Path to trained XGBoost model (.joblib)
            yolo_model_path: Path to YOLO pose model
        """
        print("Loading XGBoost model...")

        # Load trained XGBoost model
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.lag_frames = model_data["lag_frames"]
        self.keypoint_names = model_data["keypoint_names"]

        print(f"  Model type: {model_data['model_type']}")
        print(f"  Lag frames: {self.lag_frames}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        print(f"  Features: {len(self.feature_names)}")

        # Initialize YOLO
        self.yolo_model = YOLO(yolo_model_path)

        # COCO keypoint indices
        self.RELEVANT_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.RELEVANT_NAMES = self.keypoint_names

        # Label mapping
        self.label_map = {
            i: label for i, label in enumerate(self.label_encoder.classes_)
        }

        # Buffers for each tracked player
        # Each player gets a rolling buffer of (lag_frames + 1) frames
        self.player_buffers = {}
        self.last_predictions = {}

        # Track prediction cooldown to avoid spam
        self.prediction_cooldown = {}
        self.cooldown_frames = 20  # Don't predict same stroke again for 20 frames

        # Coordinate columns for feature extraction
        self.coord_cols = [
            f"{axis}_{name}" for name in self.RELEVANT_NAMES for axis in ["x", "y"]
        ]

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

    def create_lagging_features_from_buffer(self, frame_buffer):
        """
        Create lagging features from frame buffer.
        Replicates training feature engineering logic.

        Args:
            frame_buffer: List of normalized keypoint dictionaries
                         Length = lag_frames + 1
                         frame_buffer[0] = oldest, frame_buffer[-1] = current

        Returns:
            Dictionary of features matching training feature names
        """
        if len(frame_buffer) < self.lag_frames + 1:
            return None

        features = {}
        current_frame = frame_buffer[-1]

        # 1. CURRENT FRAME POSITIONS
        for col in self.coord_cols:
            features[col] = current_frame[col]

        # 2. LAGGED POSITIONS
        for lag in range(1, self.lag_frames + 1):
            past_frame = frame_buffer[-(lag + 1)]
            for col in self.coord_cols:
                features[f"{col}_lag{lag}"] = past_frame[col]

        # 3. VELOCITY FEATURES
        prev_frame = frame_buffer[-2]
        for col in self.coord_cols:
            velocity = current_frame[col] - prev_frame[col]
            features[f"{col}_velocity"] = velocity

        # Lagged velocities
        for lag in range(1, min(4, self.lag_frames + 1)):
            if len(frame_buffer) > lag + 1:
                frame_t = frame_buffer[-(lag + 1)]
                frame_t_minus_1 = frame_buffer[-(lag + 2)]
                for col in self.coord_cols:
                    velocity = frame_t[col] - frame_t_minus_1[col]
                    features[f"{col}_velocity_lag{lag}"] = velocity

        # 4. ACCELERATION FEATURES
        if len(frame_buffer) >= 3:
            frame_t = frame_buffer[-1]
            frame_t_minus_1 = frame_buffer[-2]
            frame_t_minus_2 = frame_buffer[-3]

            for col in self.coord_cols:
                v_current = frame_t[col] - frame_t_minus_1[col]
                v_previous = frame_t_minus_1[col] - frame_t_minus_2[col]
                acceleration = v_current - v_previous
                features[f"{col}_acceleration"] = acceleration

        # 5. WRIST SPEED
        for wrist in ["left_wrist", "right_wrist"]:
            vx = features[f"x_{wrist}_velocity"]
            vy = features[f"y_{wrist}_velocity"]
            speed = np.sqrt(vx**2 + vy**2)
            features[f"{wrist}_speed"] = speed

            # Lagged speeds
            for lag in range(1, min(4, self.lag_frames + 1)):
                if len(frame_buffer) > lag + 1:
                    frame_t = frame_buffer[-(lag + 1)]
                    frame_t_minus_1 = frame_buffer[-(lag + 2)]
                    vx = frame_t[f"x_{wrist}"] - frame_t_minus_1[f"x_{wrist}"]
                    vy = frame_t[f"y_{wrist}"] - frame_t_minus_1[f"y_{wrist}"]
                    speed = np.sqrt(vx**2 + vy**2)
                    features[f"{wrist}_speed_lag{lag}"] = speed

        # 6. ARM EXTENSION (elbow-wrist distance)
        for side in ["left", "right"]:
            dx = current_frame[f"x_{side}_elbow"] - current_frame[f"x_{side}_wrist"]
            dy = current_frame[f"y_{side}_elbow"] - current_frame[f"y_{side}_wrist"]
            features[f"{side}_arm_extension"] = np.sqrt(dx**2 + dy**2)

            # Lagged arm extension
            for lag in range(1, min(4, self.lag_frames + 1)):
                past_frame = frame_buffer[-(lag + 1)]
                dx = past_frame[f"x_{side}_elbow"] - past_frame[f"x_{side}_wrist"]
                dy = past_frame[f"y_{side}_elbow"] - past_frame[f"y_{side}_wrist"]
                features[f"{side}_arm_extension_lag{lag}"] = np.sqrt(dx**2 + dy**2)

        # 7. FULL ARM REACH (shoulder-wrist distance)
        for side in ["left", "right"]:
            dx = current_frame[f"x_{side}_shoulder"] - current_frame[f"x_{side}_wrist"]
            dy = current_frame[f"y_{side}_shoulder"] - current_frame[f"y_{side}_wrist"]
            features[f"{side}_full_arm_reach"] = np.sqrt(dx**2 + dy**2)

        # 8. BODY ORIENTATION
        dx = current_frame["x_left_hip"] - current_frame["x_right_hip"]
        dy = current_frame["y_left_hip"] - current_frame["y_right_hip"]
        features["hip_width"] = np.sqrt(dx**2 + dy**2)

        dx = current_frame["x_left_shoulder"] - current_frame["x_right_shoulder"]
        dy = current_frame["y_left_shoulder"] - current_frame["y_right_shoulder"]
        features["shoulder_width"] = np.sqrt(dx**2 + dy**2)

        # 9. LATERAL DOMINANCE
        left_speed = features["left_wrist_speed"]
        right_speed = features["right_wrist_speed"]
        total_speed = left_speed + right_speed

        if total_speed > 0:
            features["left_wrist_dominance"] = left_speed / total_speed
        else:
            features["left_wrist_dominance"] = 0.5

        return features

    def predict_stroke(self, player_id, frame_buffer, current_frame):
        """
        Predict stroke type from frame buffer using XGBoost

        Args:
            player_id: ID of the player
            frame_buffer: List of normalized keypoint dictionaries (rolling buffer)
            current_frame: Current frame number

        Returns:
            prediction: Stroke type (forehand/backhand/neither)
            confidence: Prediction confidence
        """
        if len(frame_buffer) < self.lag_frames + 1:
            return None, None

        # Create lagging features from buffer
        features = self.create_lagging_features_from_buffer(frame_buffer)

        if features is None:
            return None, None

        # ==================================================================================
        # FIX: Only use features that exist in the features dictionary
        # Filter out any metadata columns that got into feature_names during training
        # ==================================================================================
        valid_feature_names = [
            fname for fname in self.feature_names if fname in features
        ]

        if len(valid_feature_names) != len(self.feature_names):
            print(
                f"WARNING: Expected {len(self.feature_names)} features, but only {len(valid_feature_names)} are available"
            )
            print(
                f"Missing features: {set(self.feature_names) - set(valid_feature_names)}"
            )

        # Create feature vector in correct order (only valid features)
        try:
            feature_vector = np.array(
                [features[fname] for fname in valid_feature_names]
            ).reshape(1, -1)
        except KeyError as e:
            print(f"ERROR: Missing feature {e}")
            print(
                f"Available features: {list(features.keys())[:10]}..."
            )  # Show first 10
            return None, None

        # Get XGBoost prediction
        probabilities = self.model.predict_proba(feature_vector)[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]

        stroke_type = self.label_map[prediction]

        # Check cooldown to avoid repetitive predictions
        if player_id in self.prediction_cooldown:
            last_frame, last_stroke = self.prediction_cooldown[player_id]
            if (
                current_frame - last_frame < self.cooldown_frames
                and stroke_type == last_stroke
            ):
                return None, None

        # Only report if not "neither" and confidence > threshold
        if stroke_type != "neither" and confidence > 0.6:
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

                # Initialize buffer for new player (rolling buffer of lag_frames + 1)
                if current_id not in self.player_buffers:
                    self.player_buffers[current_id] = deque(maxlen=self.lag_frames + 1)
                    self.last_predictions[current_id] = None

                # Add to rolling buffer (automatically removes oldest when full)
                self.player_buffers[current_id].append(normalized_keypoints)

                # Predict once we have enough frames in buffer
                if len(self.player_buffers[current_id]) >= self.lag_frames + 1:
                    stroke, confidence = self.predict_stroke(
                        current_id,
                        list(self.player_buffers[current_id]),  # Convert deque to list
                        frame_idx,
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
        model_path: Path to trained XGBoost model (.joblib)
        output_path: Optional path to save output video
    """
    predictor = XGBoostStrokePredictor(model_path)

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
    all_detections = []

    print("=" * 60)
    print("Starting XGBoost inference with LAGGING FEATURES...")
    print("=" * 60)
    print(f"Lag frames: {predictor.lag_frames}")
    print(f"Cooldown: {predictor.cooldown_frames} frames")
    print("Press 'q' to quit")
    print("-" * 60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Process frame
        predictions, annotated_frame = predictor.process_frame(frame, frame_idx)

        # Store and print predictions
        for pred in predictions:
            timestamp = frame_idx / fps
            pred["timestamp"] = f"{timestamp:.2f}s"
            all_detections.append(pred)

            print(
                f"Frame {pred['frame']:5d} ({pred['timestamp']:>7s}) | "
                f"Player {pred['player_id']} did {pred['stroke'].upper():8s} "
                f"(confidence: {pred['confidence']:.3f})"
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

        # Add text overlay for last predictions
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
        cv2.imshow("XGBoost Stroke Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("-" * 60)
    print("Inference complete!")
    print(f"Total frames: {frame_idx}")
    print(f"Total detections: {len(all_detections)}")

    # Save detections to CSV
    if all_detections:
        df = pd.DataFrame(all_detections)
        csv_path = video_path.replace(".mp4", "_xgb_detections.csv")
        df.to_csv(csv_path, index=False)
        print(f"Detections saved to: {csv_path}")

        # Print summary
        print("\nDetection Summary:")
        for player in df["player_id"].unique():
            player_df = df[df["player_id"] == player]
            print(f"\nPlayer {player}:")
            print(f"  Total strokes: {len(player_df)}")
            for stroke in player_df["stroke"].unique():
                count = len(player_df[player_df["stroke"] == stroke])
                print(f"    {stroke.capitalize()}: {count}")


if __name__ == "__main__":
    # Example usage
    video_path = "/home/g03-s2025/Desktop/SquashCoachingCopilot/cv-module/digitization/event-recognition/stroke-detection/implementation/Videos/video-4.mp4"
    model_path = "/home/g03-s2025/Desktop/SquashCoachingCopilot/cv-module/digitization/event-recognition/stroke-detection/implementation/models/stroke_xgb_model.joblib"
    output_path = "/home/g03-s2025/Desktop/SquashCoachingCopilot/cv-module/digitization/event-recognition/stroke-detection/implementation/Videos/output_xgb_video.mp4"

    run_inference(video_path, model_path, output_path)
