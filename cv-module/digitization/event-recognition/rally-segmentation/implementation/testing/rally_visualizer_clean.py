"""
Clean Rally Visualizer Pipeline
Real-time rally state visualization using unified prediction interface.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict
from collections import deque

from utilities.metrics_aggregator import MetricsAggregator
from modeling.unified_predictor import UnifiedPredictor
from utilities.general import load_config


class RallyVisualizer:
    """
    Clean rally state visualizer using unified prediction interface.
    Model-agnostic visualization pipeline.
    """

    def __init__(self):
        """Initialize the rally visualizer."""
        self.config = load_config()
        self.window_size = self.config["annotations"]["window_size"]
        self.show_metrics = self.config["testing"]["show_metrics"]

        # Initialize metrics aggregator for live feature extraction
        self.metrics_aggregator = MetricsAggregator(window_size=self.window_size)
        self.current_metrics = None

        # Initialize unified predictor (model-agnostic)
        print("Initializing unified predictor...")
        self.predictor = UnifiedPredictor()
        print(f"Loaded: {self.predictor.get_model_info()}")

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

    def _predict_state(self, metrics: Dict[str, any]) -> str:
        """
        Predict state using unified predictor (model-agnostic).

        Args:
            metrics: Base metrics from MetricsAggregator

        Returns:
            Predicted state: "start", "active", or "end"
        """
        try:
            # Use unified predictor for consistent results
            predicted_state = self.predictor.predict(metrics)
            return predicted_state

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Simple fallback based on distance
            distance = metrics.get("mean_distance", 0)
            if distance < 2.5:
                return "active"
            elif distance > 4.0:
                return "end"
            else:
                return "start"

    def _draw_state_banner(self, frame: np.ndarray, state: str) -> np.ndarray:
        """Draw state banner at the top of the frame."""
        height, width = frame.shape[:2]
        banner_height = 80

        color = self.state_colors.get(state, (128, 128, 128))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # Get model info for display
        model_info = self.predictor.get_model_info()
        model_display = model_info["model_type"].replace("_", " ").title()

        state_text = f"STATE: {state.upper()} ({model_display})"
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

        # Get model info
        model_info = self.predictor.get_model_info()

        metrics = [
            f"Frame: {frame_num}",
            f"Window: {metrics_history_len}/{self.window_size}",
            "",
            f"State: {state.upper()}",
            f"Model: {model_info['model_type'].replace('_', ' ').title()}",
            "",
            f"--- Metrics ---",
            f"Player 1: ({safe_format(state_metrics.get('median_player1_x'))}, {safe_format(state_metrics.get('median_player1_y'))})",
            f"Player 2: ({safe_format(state_metrics.get('median_player2_x'))}, {safe_format(state_metrics.get('median_player2_y'))})",
            f"Distance: {safe_format(state_metrics.get('mean_distance'))} m",
        ]

        # Add ML-specific info if applicable
        if model_info["model_type"] == "ml_based":
            metrics.extend(
                [
                    "",
                    f"--- ML Info ---",
                    f"Features: {model_info.get('num_features', 'N/A')}",
                    f"ML Type: {model_info.get('ml_model_type', 'N/A')}",
                ]
            )

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
        """Process video with unified state prediction."""
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
        print(f"RALLY STATE VISUALIZATION")
        print(f"{'='*60}")
        print(f"\nVideo Info:")
        print(f"  File: {Path(self.config['testing']['video_path']).name}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Processing: {start_frame} to {end_frame}")
        print(f"  Window Size: {self.window_size} frames")

        model_info = self.predictor.get_model_info()
        print(f"  Model: {model_info['model_type'].replace('_', ' ').title()}")
        if model_info["model_type"] == "ml_based":
            print(f"    Type: {model_info.get('ml_model_type', 'Unknown')}")
            print(f"    Features: {model_info.get('num_features', 'Unknown')}")

        # Reset predictor state for new video
        self.predictor.reset_state()

        writer = None
        output_path = None
        if self.config["testing"]["visualizer_output_directory"]:
            model_suffix = model_info["model_type"]
            output_path = (
                self.config["testing"]["visualizer_output_directory"]
                + "/"
                + f"{Path(self.config['testing']['video_path']).stem}_{model_suffix}_output.mp4"
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
        current_state = "start"  # Initialize display state

        try:
            while frame_num < end_frame:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Update metrics aggregator
                    self.metrics_aggregator.update_metrics(frame, frame_num)

                    # Process with unified predictor when we have a full window
                    if self.metrics_aggregator.has_full_window():
                        self.current_metrics = (
                            self.metrics_aggregator.get_aggregated_metrics()
                        )

                        # Predict state using unified predictor (model-agnostic)
                        predicted_state = self._predict_state(self.current_metrics)
                        current_state = predicted_state

                        self.state_history.append(current_state)
                        self.frame_history.append(frame_num)

                    # Create visualization
                    vis_frame = frame.copy()
                    player_positions = self.metrics_aggregator.last_player_bboxes
                    vis_frame = self._draw_player_tracking(vis_frame, player_positions)
                    vis_frame = self._draw_state_banner(vis_frame, current_state)

                    if self.show_metrics and self.current_metrics:
                        vis_frame = self._draw_metrics_panel(
                            vis_frame,
                            frame_num,
                            current_state,
                            self.current_metrics,
                        )

                    cv2.imshow("Rally State Visualization", vis_frame)

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
                            f"Progress: {progress:.1f}% - Frame {frame_num}/{end_frame} - State: {current_state}",
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

        # Initialize MetricsAggregator with court calibration and tracking
        self.metrics_aggregator.calibrate_court(first_frame)
        self.metrics_aggregator.initialize_tracker(first_frame)

        print("Starting video processing...")
        self.process_video()


if __name__ == "__main__":
    try:
        visualizer = RallyVisualizer()
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")
        raise
