import cv2
import numpy as np
from pathlib import Path
from typing import Dict
from collections import deque

from utilities.metrics_aggregator import MetricsAggregator
from modeling.rally_state_segmenter import RallyStateSegmenter
from utilities.general import load_config


class RallyVisualizer:
    """
    Real-time rally state visualizer integrated with player tracking and court calibration.
    Processes video, tracks players, calculates distances, and displays rally states.
    """

    def __init__(self):
        """Initialize the integrated visualizer."""
        self.config = load_config()
        self.window_size = self.config["annotations"]["window_size"]
        self.show_metrics = self.config["testing"]["show_metrics"]
        self.show_timeline = self.config["testing"]["show_timeline"]
        self.show_court_view = self.config["testing"]["show_court"]

        # Initialize metrics aggregator
        self.metrics_aggregator = MetricsAggregator(window_size=self.window_size)

        # State colors (BGR)
        self.state_colors = {
            "start": (255, 200, 0),  # Cyan
            "active": (0, 255, 0),  # Green
            "end": (0, 0, 255),  # Red
        }

        # Player colors (BGR)
        self.player_colors = {1: (0, 255, 0), 2: (0, 0, 255)}

        # History for visualization
        self.distance_history = deque(maxlen=300)
        self.state_history = deque(maxlen=300)
        self.frame_history = deque(maxlen=300)

        # Current state
        self.current_state = "start"
        self.frames_in_state = 0

    def _predict_state(self) -> str:
        """Predict current state based on aggregated distance."""
        if not self.metrics_aggregator.has_full_window():
            return self.current_state

        mean_distance = self.metrics_aggregator.get_mean_distance()
        state = self.model._classify_frame_distance_only(
            mean_distance, self.current_state
        )

        if state == self.current_state:
            self.frames_in_state += 1
        else:
            if self.model._check_min_duration(self.current_state, self.frames_in_state):
                if self.model._is_valid_transition(self.current_state, state):
                    self.current_state = state
                    self.frames_in_state = 1
            else:
                self.frames_in_state += 1

        return self.current_state

    def _draw_state_banner(self, frame: np.ndarray, state: str) -> np.ndarray:
        """Draw state banner at the top of the frame."""
        height, width = frame.shape[:2]
        banner_height = 80

        color = self.state_colors.get(state, (128, 128, 128))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        state_text = f"STATE: {state.upper()}"
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
            # bbox should be a tuple of (x1, y1, x2, y2)
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
        self, frame: np.ndarray, frame_num: int, state: str, mean_distance: float
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
        metrics = [
            f"Frame: {frame_num}",
            f"Window: {metrics_history_len}/{self.window_size}",
            "",
            f"State: {state.upper()}",
            f"Duration: {self.frames_in_state} frames",
            "",
            f"Mean Distance: {mean_distance:.2f}m",
            "",
            "--- Thresholds ---",
        ]

        active_min, active_max = tuple(self.model.config["distance_active_range"])
        start_min, start_max = tuple(self.model.config["distance_start_range"])
        end_min, _ = tuple(self.model.config["distance_end_range"])

        metrics.extend(
            [
                f"Active: {active_min:.1f}-{active_max:.1f}m",
                f"Start: {start_min:.1f}-{start_max:.1f}m",
                f"End: >{end_min:.1f}m",
            ]
        )

        for i, text in enumerate(metrics):
            y = y_offset + i * line_height
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

    def _draw_distance_timeline(self, frame: np.ndarray) -> np.ndarray:
        """Draw distance timeline at the bottom."""
        if len(self.distance_history) < 2:
            return frame

        height, width = frame.shape[:2]
        plot_height = 150
        plot_y_start = height - plot_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, plot_y_start), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        distances = list(self.distance_history)
        states = list(self.state_history)

        max_distance = max(distances) if distances else 8.0
        min_distance = 0.0

        for i in range(5):
            y = plot_y_start + int((plot_height * i) / 4)
            cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)

        def dist_to_y(d):
            normalized = (d - min_distance) / (max_distance - min_distance + 1e-6)
            return plot_y_start + plot_height - int(normalized * plot_height)

        active_min, active_max = tuple(self.model.config["distance_active_range"])
        start_min, start_max = tuple(self.model.config["distance_start_range"])

        y_active = dist_to_y(active_max)
        cv2.line(frame, (0, y_active), (width, y_active), (0, 255, 0), 2, cv2.LINE_AA)

        y_start = dist_to_y(start_max)
        cv2.line(frame, (0, y_start), (width, y_start), (255, 200, 0), 2, cv2.LINE_AA)

        points = []
        for i, distance in enumerate(distances):
            x = int((i / len(distances)) * width)
            y = dist_to_y(distance)
            points.append((x, y))

        for i in range(len(points) - 1):
            state = states[i] if i < len(states) else "start"
            color = self.state_colors.get(state, (128, 128, 128))
            cv2.line(frame, points[i], points[i + 1], color, 2, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            f"Distance Timeline ({self.window_size}-frame avg)",
            (10, plot_y_start + 25),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    def _draw_court_view(
        self, frame: np.ndarray, player_real_positions: Dict
    ) -> np.ndarray:
        """Draw top-down court view in corner."""
        real_coords = self.metrics_aggregator.calibrator.config.get("real_coords", [])
        if len(real_coords) < 4:
            return frame

        xs = [coord[0] for coord in real_coords]
        ys = [coord[1] for coord in real_coords]
        court_min_x, court_max_x = min(xs), max(xs)
        court_min_y, court_max_y = min(ys), max(ys)
        court_width = court_max_x - court_min_x
        court_length = court_max_y - court_min_y

        view_width = 250
        view_height = int(view_width * (court_length / court_width))

        margin = 20
        view_x = frame.shape[1] - view_width - margin
        view_y = 100

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (view_x, view_y),
            (view_x + view_width, view_y + view_height),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            overlay,
            (view_x, view_y),
            (view_x + view_width, view_y + view_height),
            (255, 255, 255),
            2,
        )

        center_x = view_x + view_width // 2
        cv2.line(
            overlay,
            (center_x, view_y),
            (center_x, view_y + view_height),
            (255, 255, 255),
            1,
        )

        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        for player_id in [1, 2]:
            if (
                player_id in player_real_positions
                and len(player_real_positions[player_id]) > 0
            ):
                real_pos = player_real_positions[player_id][-1]
                if real_pos:
                    normalized_x = (real_pos[0] - court_min_x) / court_width
                    normalized_y = (real_pos[1] - court_min_y) / court_length

                    px = int(view_x + normalized_x * view_width)
                    py = int(view_y + normalized_y * view_height)

                    color = self.player_colors.get(player_id, (255, 255, 255))
                    cv2.circle(frame, (px, py), 8, color, -1)
                    cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "Court View",
            (view_x, view_y - 5),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    def process_video(self):
        """Process video with integrated tracking and state prediction."""
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
        print(f"INTEGRATED RALLY STATE VISUALIZATION")
        print(f"{'='*60}")
        print(f"\nVideo Info:")
        print(f"  File: {Path(self.config['testing']['video_path']).name}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Processing: {start_frame} to {end_frame}")
        print(f"  Window Size: {self.window_size} frames")

        writer = None
        output_path = None
        if self.config["testing"]["visualizer_output_directory"]:
            output_path = (
                self.config["testing"]["visualizer_output_directory"]
                + "/"
                + f"{Path(self.config['testing']['video_path']).stem}_output.mp4"
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

                    positions = self.metrics_aggregator.get_player_positions()
                    player_positions = self.metrics_aggregator.last_player_bboxes

                    player_real_positions = positions.get("real", {})
                    mean_distance = self.metrics_aggregator.get_mean_distance()

                    if self.metrics_aggregator.has_full_window():
                        state = self._predict_state()

                        self.distance_history.append(mean_distance)
                        self.state_history.append(state)
                        self.frame_history.append(frame_num)

                    else:
                        state = "start"

                    vis_frame = frame.copy()
                    vis_frame = self._draw_player_tracking(vis_frame, player_positions)
                    vis_frame = self._draw_state_banner(vis_frame, state)

                    if self.show_metrics:
                        vis_frame = self._draw_metrics_panel(
                            vis_frame, frame_num, state, mean_distance
                        )

                    if self.show_timeline:
                        vis_frame = self._draw_distance_timeline(vis_frame)

                    if self.show_court_view:
                        vis_frame = self._draw_court_view(
                            vis_frame, player_real_positions
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
                            f"Progress: {progress:.1f}% - Frame {frame_num}/{end_frame} - State: {state}",
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

        print("Initializing segmentation model...")
        self.model = RallyStateSegmenter()

        print("Starting video processing...")
        self.process_video()


if __name__ == "__main__":
    visualizer = RallyVisualizer()
    visualizer.run()
