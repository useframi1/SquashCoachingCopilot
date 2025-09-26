import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from collections import deque

from utilities.metrics_aggregator import MetricsAggregator
from modeling.rally_state_predictor import RallyStatePredictor  # Import your ML model
from utilities.general import load_config


class RallyVisualizer:
    """
    Real-time rally state visualizer integrated with player tracking and court calibration.
    Uses trained ML model for state prediction instead of rule-based approach.
    """

    def __init__(self, model_path: str = "rally_state_model.pkl"):
        """Initialize the integrated visualizer with ML model."""
        self.config = load_config()
        self.window_size = self.config["annotations"]["window_size"]
        self.show_metrics = self.config["testing"]["show_metrics"]

        # Initialize metrics aggregator with feature engineering
        self.metrics_aggregator = MetricsAggregator(window_size=self.window_size)
        self.current_metrics = None

        # Load trained ML model
        print(f"Loading ML model from {model_path}...")
        try:
            self.ml_model = RallyStatePredictor.load_model(model_path)
            print("ML model loaded successfully!")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            print("Please ensure the model file exists and was trained properly.")
            raise

        # State colors (BGR)
        self.state_colors = {
            "start": (255, 200, 0),  # Cyan
            "active": (0, 255, 0),  # Green
            "end": (0, 0, 255),  # Red
        }

        # Player colors (BGR)
        self.player_colors = {1: (0, 255, 0), 2: (0, 0, 255)}

        # History for visualization
        self.state_history = deque(maxlen=300)
        self.frame_history = deque(maxlen=300)

        # Track state for feature engineering
        self.current_state = "start"  # Initialize with start state
        self.state_duration = 1
        self.frames_since_state_change = 0

        # Store previous metrics for feature engineering
        self.previous_metrics = []

        # Buffer for end state confirmation
        self.end_prediction_buffer = []
        self.end_confirmation_frames = 5  # Require 5 consecutive predictions

    def _engineer_features(self, base_metrics: Dict[str, any]) -> Dict[str, any]:
        """
        Engineer features needed for ML model prediction.

        Args:
            base_metrics: Basic metrics from aggregator

        Returns:
            Enhanced metrics with engineered features
        """
        enhanced_metrics = base_metrics.copy()

        # Store current metrics for feature engineering
        self.previous_metrics.append(base_metrics)

        # Keep only last 4 metrics for lag features (current + 3 lags)
        if len(self.previous_metrics) > 4:
            self.previous_metrics = self.previous_metrics[-4:]

        # Distance lag features
        if len(self.previous_metrics) >= 2:
            enhanced_metrics["distance_lag_1"] = self.previous_metrics[-2].get(
                "mean_distance", 0
            )
        else:
            enhanced_metrics["distance_lag_1"] = enhanced_metrics.get(
                "mean_distance", 0
            )

        if len(self.previous_metrics) >= 3:
            enhanced_metrics["distance_lag_2"] = self.previous_metrics[-3].get(
                "mean_distance", 0
            )
        else:
            enhanced_metrics["distance_lag_2"] = enhanced_metrics.get(
                "mean_distance", 0
            )

        if len(self.previous_metrics) >= 4:
            enhanced_metrics["distance_lag_3"] = self.previous_metrics[-4].get(
                "mean_distance", 0
            )
        else:
            enhanced_metrics["distance_lag_3"] = enhanced_metrics.get(
                "mean_distance", 0
            )

        # Distance change and acceleration
        if len(self.previous_metrics) >= 2:
            current_dist = enhanced_metrics.get("mean_distance", 0)
            prev_dist = enhanced_metrics["distance_lag_1"]
            enhanced_metrics["distance_change"] = current_dist - prev_dist

            if len(self.previous_metrics) >= 3:
                prev_change = prev_dist - enhanced_metrics["distance_lag_2"]
                enhanced_metrics["distance_acceleration"] = (
                    enhanced_metrics["distance_change"] - prev_change
                )
            else:
                enhanced_metrics["distance_acceleration"] = 0
        else:
            enhanced_metrics["distance_change"] = 0
            enhanced_metrics["distance_acceleration"] = 0

        # Rolling statistics (using recent 3 frames)
        recent_distances = [
            m.get("mean_distance", 0)
            for m in self.previous_metrics[-3:]
            if m.get("mean_distance") is not None
        ]
        if recent_distances:
            enhanced_metrics["distance_rolling_mean"] = np.mean(recent_distances)
            enhanced_metrics["distance_rolling_std"] = (
                np.std(recent_distances) if len(recent_distances) > 1 else 0
            )
            enhanced_metrics["distance_rolling_min"] = np.min(recent_distances)
            enhanced_metrics["distance_rolling_max"] = np.max(recent_distances)
        else:
            enhanced_metrics["distance_rolling_mean"] = enhanced_metrics.get(
                "mean_distance", 0
            )
            enhanced_metrics["distance_rolling_std"] = 0
            enhanced_metrics["distance_rolling_min"] = enhanced_metrics.get(
                "mean_distance", 0
            )
            enhanced_metrics["distance_rolling_max"] = enhanced_metrics.get(
                "mean_distance", 0
            )

        # Player movement features
        if len(self.previous_metrics) >= 2:
            prev_metrics = self.previous_metrics[-2]

            # Player 1 movement
            p1_x_curr = enhanced_metrics.get("median_player1_x", 0)
            p1_y_curr = enhanced_metrics.get("median_player1_y", 0)
            p1_x_prev = prev_metrics.get("median_player1_x", p1_x_curr)
            p1_y_prev = prev_metrics.get("median_player1_y", p1_y_curr)
            enhanced_metrics["player1_movement"] = np.sqrt(
                (p1_x_curr - p1_x_prev) ** 2 + (p1_y_curr - p1_y_prev) ** 2
            )

            # Player 2 movement
            p2_x_curr = enhanced_metrics.get("median_player2_x", 0)
            p2_y_curr = enhanced_metrics.get("median_player2_y", 0)
            p2_x_prev = prev_metrics.get("median_player2_x", p2_x_curr)
            p2_y_prev = prev_metrics.get("median_player2_y", p2_y_curr)
            enhanced_metrics["player2_movement"] = np.sqrt(
                (p2_x_curr - p2_x_prev) ** 2 + (p2_y_curr - p2_y_prev) ** 2
            )
        else:
            enhanced_metrics["player1_movement"] = 0
            enhanced_metrics["player2_movement"] = 0

        # Court position features (using config values)
        court_center_x = 3.2  # From your config
        service_line_y = 5.44  # From your config

        p1_x = enhanced_metrics.get("median_player1_x", court_center_x)
        p2_x = enhanced_metrics.get("median_player2_x", court_center_x)
        p1_y = enhanced_metrics.get("median_player1_y", service_line_y)
        p2_y = enhanced_metrics.get("median_player2_y", service_line_y)

        enhanced_metrics["player1_court_side"] = 1 if p1_x > court_center_x else 0
        enhanced_metrics["player2_court_side"] = 1 if p2_x > court_center_x else 0
        enhanced_metrics["players_same_side"] = (
            1
            if enhanced_metrics["player1_court_side"]
            == enhanced_metrics["player2_court_side"]
            else 0
        )

        enhanced_metrics["player1_from_service_line"] = p1_y - service_line_y
        enhanced_metrics["player2_from_service_line"] = p2_y - service_line_y

        # Previous state and temporal features
        enhanced_metrics["prev_state"] = self.current_state
        enhanced_metrics["state_duration"] = self.state_duration
        enhanced_metrics["frames_since_state_change"] = self.frames_since_state_change

        return enhanced_metrics

    def _predict_state(self, metrics: Dict[str, any]) -> str:
        """Predict current state using trained ML model."""
        try:
            # Engineer features for ML model
            enhanced_metrics = self._engineer_features(metrics)

            # Create DataFrame with single row for prediction
            df = pd.DataFrame([enhanced_metrics])

            # Ensure we have all required features (fill missing with defaults)
            required_features = self.ml_model.feature_names
            for feature in required_features:
                if feature not in df.columns:
                    if "distance" in feature:
                        df[feature] = 0.0
                    elif "player" in feature and ("x" in feature or "y" in feature):
                        df[feature] = 0.0
                    elif "movement" in feature:
                        df[feature] = 0.0
                    elif "side" in feature:
                        df[feature] = 0
                    else:
                        df[feature] = 0

            # Encode prev_state
            if "prev_state_encoded" in required_features:
                try:
                    df["prev_state_encoded"] = (
                        self.ml_model.prev_state_encoder.transform(
                            [enhanced_metrics["prev_state"]]
                        )[0]
                    )
                except:
                    df["prev_state_encoded"] = 0  # Default encoding

            # Remove non-feature columns
            features_to_keep = [col for col in df.columns if col in required_features]
            prediction_df = df[features_to_keep]

            # Make raw prediction
            raw_prediction = self.ml_model.predict(prediction_df)[0]

            # Apply constraints to make end state harder to reach
            predicted_state = self._apply_end_state_constraints(
                raw_prediction, enhanced_metrics
            )

            # Update state tracking
            if predicted_state != self.current_state:
                self.frames_since_state_change = 0
                self.state_duration = 1
            else:
                self.state_duration += 1

            self.frames_since_state_change += 1
            self.current_state = predicted_state

            return predicted_state

        except Exception as e:
            print(f"Error in ML prediction: {e}")
            # Fallback to simple distance-based logic
            distance = metrics.get("mean_distance", 0)
            if distance < 2.5:
                return "active"
            elif distance > 4.0:
                return "end"
            else:
                return "start"

    def _apply_end_state_constraints(
        self, raw_prediction: str, enhanced_metrics: Dict[str, any]
    ) -> str:
        """
        Apply constraints to make it harder to predict end state.
        Uses both rule-based constraints and confirmation buffer.

        Args:
            raw_prediction: Original model prediction
            enhanced_metrics: Feature dictionary

        Returns:
            Constrained prediction
        """
        # If not predicting end, clear buffer and return as-is
        if raw_prediction != "end":
            self.end_prediction_buffer.clear()
            return raw_prediction

        # Apply rule-based constraints first
        distance = enhanced_metrics.get("mean_distance", 0)
        distance_change = enhanced_metrics.get("distance_change", 0)

        # # Stricter distance threshold
        # if distance < 3.5:  # Higher than original 4.0
        #     self.end_prediction_buffer.clear()
        #     return self.current_state

        # # Require positive distance trend (players separating)
        # if distance_change < 0.1:  # Must be increasing
        #     self.end_prediction_buffer.clear()
        #     return self.current_state

        # # Don't allow end immediately after start
        # if self.current_state == "start":
        #     self.end_prediction_buffer.clear()
        #     return "active"  # Force through active state

        # # Require minimum time in active state
        # if self.current_state == "active" and self.state_duration < 10:
        #     self.end_prediction_buffer.clear()
        #     return "active"

        # If rule-based constraints pass, use buffer confirmation
        self.end_prediction_buffer.append("end")

        if len(self.end_prediction_buffer) >= self.end_confirmation_frames:
            # Log when end state is actually confirmed (for debugging)
            print(
                f"END STATE CONFIRMED after {len(self.end_prediction_buffer)} consecutive predictions - Distance: {distance:.2f}, Change: {distance_change:.2f}"
            )
            return "end"
        else:
            # Still waiting for confirmation
            print(
                f"END buffered ({len(self.end_prediction_buffer)}/{self.end_confirmation_frames}) - Distance: {distance:.2f}"
            )
            return self.current_state  # Stay in current state

    def _draw_state_banner(self, frame: np.ndarray, state: str) -> np.ndarray:
        """Draw state banner at the top of the frame."""
        height, width = frame.shape[:2]
        banner_height = 80

        color = self.state_colors.get(state, (128, 128, 128))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        state_text = f"STATE: {state.upper()} (ML)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3

        (text_width, text_height), _ = cv2.getTextSize(
            state_text, font, font_scale, thickness
        )
        text_x = (width - text_width) // 2
        text_y = (banner_height + text_height) // 2

        cv2.putText(
            frame,
            state_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            state_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        return frame

    def _draw_player_tracking(
        self, frame: np.ndarray, player_positions: Dict
    ) -> np.ndarray:
        """Draw player bounding boxes and IDs."""
        for player_id, bbox in player_positions.items():
            try:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            except (TypeError, ValueError) as e:
                print(
                    f"Warning: Invalid bbox for player {player_id}: {bbox}, error: {e}"
                )
                continue

            color = self.player_colors.get(player_id, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"P{player_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (x1, y1 - 10), font, 0.8, color, 2, cv2.LINE_AA)

        return frame

    def _draw_metrics_panel(
        self,
        frame: np.ndarray,
        frame_num: int,
        state: str,
        state_metrics: Dict[str, any],
    ) -> np.ndarray:
        """Draw metrics panel on the right side."""
        height, width = frame.shape[:2]
        panel_width = 350
        panel_x = width - panel_width

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 100), (width, height - 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        y_offset = 140
        line_height = 35

        metrics_history_len = len(self.metrics_aggregator.metrics_history)

        # Format positions safely
        def safe_format(value, default="N/A"):
            if value is not None and not np.isnan(value):
                return f"{float(value):.1f}"
            return default

        metrics = [
            f"Frame: {frame_num}",
            f"Window: {metrics_history_len}/{self.window_size}",
            "",
            f"State: {state.upper()} (ML Model)",
            f"Duration: {self.state_duration} frames",
            "",
            f"--- Metrics ---",
            f"Player 1: ({safe_format(state_metrics.get('median_player1_x'))}, {safe_format(state_metrics.get('median_player1_y'))})",
            f"Player 2: ({safe_format(state_metrics.get('median_player2_x'))}, {safe_format(state_metrics.get('median_player2_y'))})",
            f"Distance: {safe_format(state_metrics.get('mean_distance'))} m",
            "",
            f"--- ML Features ---",
            f"Prev State: {self.current_state}",
            f"Distance Change: {safe_format(state_metrics.get('distance_change', 0))}",
            f"Player Movement: {safe_format(state_metrics.get('player1_movement', 0))}",
        ]

        for i, text in enumerate(metrics):
            y = y_offset + i * line_height
            if y < height - 50:  # Don't draw beyond frame
                cv2.putText(
                    frame,
                    text,
                    (panel_x + 10, y),
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        return frame

    def process_video(self):
        """Process video with integrated tracking and ML state prediction."""
        cap = cv2.VideoCapture(self.config["testing"]["video_path"])
        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {self.config['testing']['video_path']}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = self.config["testing"]["start_frame"]
        end_frame = self.config["testing"]["end_frame"]

        if end_frame == -1:
            end_frame = total_frames

        print(f"\n{'='*60}")
        print(f"ML-BASED RALLY STATE VISUALIZATION")
        print(f"{'='*60}")
        print(f"\nVideo Info:")
        print(f"  File: {Path(self.config['testing']['video_path']).name}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Processing: {start_frame} to {end_frame}")
        print(f"  Window Size: {self.window_size} frames")
        print(
            f"  Model: {self.ml_model.model_type} with {len(self.ml_model.feature_names)} features"
        )

        writer = None
        output_path = None
        if self.config["testing"]["visualizer_output_directory"]:
            output_path = (
                self.config["testing"]["visualizer_output_directory"]
                + "/"
                + f"{Path(self.config['testing']['video_path']).stem}_ml_output.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (frame_width, frame_height)
            )
            print(f"  Output: {output_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        delay = max(1, int((1000 / fps) / self.config["testing"]["playback_speed"]))

        print(f"\nControls:")
        print(f"  SPACE - Pause/Resume")
        print(f"  Q/ESC - Quit")
        print(f"\nProcessing...\n")

        frame_num = start_frame
        paused = False

        try:
            while frame_num < end_frame:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    self.metrics_aggregator.update_metrics(frame, frame_num)

                    if self.metrics_aggregator.has_full_window():
                        self.current_metrics = (
                            self.metrics_aggregator.get_aggregated_metrics()
                        )

                        predicted_state = self._predict_state(self.current_metrics)
                        self.state_history.append(predicted_state)
                        self.frame_history.append(frame_num)

                    vis_frame = frame.copy()
                    player_positions = self.metrics_aggregator.last_player_bboxes
                    vis_frame = self._draw_player_tracking(vis_frame, player_positions)
                    vis_frame = self._draw_state_banner(vis_frame, self.current_state)

                    if self.show_metrics and self.current_metrics:
                        vis_frame = self._draw_metrics_panel(
                            vis_frame,
                            frame_num,
                            self.current_state,
                            self.current_metrics,
                        )

                    cv2.imshow("ML Rally State Visualization", vis_frame)

                    if writer:
                        writer.write(vis_frame)

                key = cv2.waitKey(delay if not paused else 1) & 0xFF

                if key == ord("q") or key == 27:
                    print("\nStopping...")
                    break
                elif key == ord(" "):
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")

                if not paused:
                    frame_num += 1

                    if frame_num % 50 == 0:
                        progress = (
                            (frame_num - start_frame) / (end_frame - start_frame)
                        ) * 100
                        print(
                            f"Progress: {progress:.1f}% - Frame {frame_num}/{end_frame} - State: {self.current_state}",
                            end="\r",
                        )

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            print(f"\n\nProcessing complete!")
            print(f"Processed {frame_num - start_frame} frames")

            if output_path:
                print(f"Video saved to: {output_path}")

    def run(self):
        """Initialize components and run the visualizer."""
        print("Initializing court calibrator...")

        cap = cv2.VideoCapture(self.config["testing"]["video_path"])
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read first frame")

        self.metrics_aggregator.calibrate_court(first_frame)
        self.metrics_aggregator.initialize_tracker(first_frame)

        print("Starting video processing with ML model...")
        self.process_video()


if __name__ == "__main__":
    # You can specify a different model path if needed
    visualizer = RallyVisualizer(model_path="models/rally_state_model.pkl")
    visualizer.run()
