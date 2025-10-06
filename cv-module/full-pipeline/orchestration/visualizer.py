"""Visualization module for rendering pipeline results on frames."""

import cv2
import numpy as np
from typing import Optional, Dict
from data.data_models import FrameData, PlayerData, BallData


class Visualizer:
    """
    Handles all visualization and rendering logic.

    Responsibilities:
    - Draw bounding boxes, keypoints, and tracking information
    - Render court keypoints and calibration data
    - Display rally state and other metadata
    - Completely decoupled from pipeline orchestration
    """

    # COCO keypoint skeleton connections (used by YOLO pose models)
    SKELETON = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],  # Head/shoulders
        [6, 12],
        [7, 13],
        [6, 7],  # Torso
        [6, 8],
        [8, 10],
        [7, 9],
        [9, 11],  # Arms
        [12, 14],
        [14, 16],
        [13, 15],
        [15, 17],  # Legs
    ]

    def __init__(
        self,
        show_court_keypoints: bool = True,
        show_player_keypoints: bool = True,
        show_player_bbox: bool = True,
        show_ball: bool = True,
        show_rally_state: bool = True,
        show_stroke_type: bool = True,
        keypoint_confidence_threshold: float = 0.5,
    ):
        """
        Initialize visualizer.

        Args:
            show_court_keypoints: Whether to display court keypoints
            show_player_keypoints: Whether to display player pose keypoints
            show_player_bbox: Whether to display player bounding boxes
            show_ball: Whether to display ball position
            show_rally_state: Whether to display rally state
            show_stroke_type: Whether to display stroke type for both players
            keypoint_confidence_threshold: Minimum confidence for displaying keypoints
        """
        self.show_court_keypoints = show_court_keypoints
        self.show_player_keypoints = show_player_keypoints
        self.show_player_bbox = show_player_bbox
        self.show_ball = show_ball
        self.show_rally_state = show_rally_state
        self.show_stroke_type = show_stroke_type
        self.keypoint_confidence_threshold = keypoint_confidence_threshold

        # Colors
        self.player_colors = {
            1: (0, 255, 0),  # Green for Player 1
            2: (255, 0, 0),  # Blue for Player 2
        }
        self.ball_color = (0, 0, 255)  # Red for ball
        self.court_color = (0, 255, 255)  # Yellow for court

    def render_frame(self, frame: np.ndarray, frame_data: FrameData) -> np.ndarray:
        """
        Render all visualization elements on a frame.

        Args:
            frame: Input frame
            frame_data: Processed frame data

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        # Draw court keypoints (only on first frame or if requested)
        if self.show_court_keypoints and frame_data.court.is_calibrated:
            annotated_frame = self._draw_court_keypoints(
                annotated_frame, frame_data.court.keypoints
            )

        # Draw player tracking
        if self.show_player_bbox or self.show_player_keypoints:
            annotated_frame = self._draw_player(annotated_frame, frame_data.player1)
            annotated_frame = self._draw_player(annotated_frame, frame_data.player2)

        # Draw ball
        if self.show_ball:
            annotated_frame = self._draw_ball(annotated_frame, frame_data.ball)

        # Draw rally state
        if self.show_rally_state:
            annotated_frame = self._draw_rally_state(
                annotated_frame, frame_data.rally_state
            )

        # Draw stroke types
        if self.show_stroke_type:
            annotated_frame = self._draw_stroke_types(
                annotated_frame, frame_data.player1, frame_data.player2
            )

        return annotated_frame

    def _draw_court_keypoints(
        self, frame: np.ndarray, keypoints: Optional[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Draw court keypoints on frame."""
        if keypoints is None:
            return frame

        for class_name, kp_array in keypoints.items():
            for i, (x, y) in enumerate(kp_array):
                cv2.circle(frame, (int(x), int(y)), 3, self.court_color, -1)

        return frame

    def _draw_player(self, frame: np.ndarray, player_data: PlayerData) -> np.ndarray:
        """Draw player tracking results (bbox, keypoints, skeleton)."""
        if not player_data.is_valid():
            return frame

        player_id = player_data.player_id
        color = self.player_colors.get(player_id, (255, 255, 255))

        x1, y1, x2, y2 = map(int, player_data.bbox)

        # Draw bounding box
        if self.show_player_bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw position (bottom-center of bbox)
        pos = player_data.position
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, color, -1)

        # Draw label with confidence
        if player_data.confidence is not None:
            label = f"P{player_id}: {player_data.confidence:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Draw keypoints and skeleton
        if self.show_player_keypoints and player_data.keypoints:
            frame = self._draw_player_keypoints(frame, player_data, color)

        return frame

    def _draw_player_keypoints(
        self, frame: np.ndarray, player_data: PlayerData, color: tuple
    ) -> np.ndarray:
        """Draw player pose keypoints and skeleton."""
        keypoints = player_data.keypoints
        if not keypoints or not keypoints.get("xy"):
            return frame

        kp_array = keypoints["xy"]
        conf_array = keypoints.get("conf")

        # Draw skeleton lines first (so they appear behind keypoints)
        if conf_array:
            for joint1, joint2 in self.SKELETON:
                # COCO uses 1-based indexing, convert to 0-based
                idx1, idx2 = joint1 - 1, joint2 - 1
                if idx1 < len(kp_array) and idx2 < len(kp_array):
                    if (
                        conf_array[idx1] > self.keypoint_confidence_threshold
                        and conf_array[idx2] > self.keypoint_confidence_threshold
                    ):
                        pt1 = (int(kp_array[idx1][0]), int(kp_array[idx1][1]))
                        pt2 = (int(kp_array[idx2][0]), int(kp_array[idx2][1]))
                        cv2.line(frame, pt1, pt2, color, 2)

        # Draw keypoints on top
        for i, (kp_x, kp_y) in enumerate(kp_array):
            if conf_array and conf_array[i] > self.keypoint_confidence_threshold:
                cv2.circle(frame, (int(kp_x), int(kp_y)), 3, color, -1)

        return frame

    def _draw_ball(self, frame: np.ndarray, ball_data: BallData) -> np.ndarray:
        """Draw ball detection results."""
        if not ball_data.is_valid():
            return frame

        ball_x, ball_y = ball_data.position
        cv2.circle(frame, (ball_x, ball_y), 5, self.ball_color, -1)
        cv2.putText(
            frame,
            "Ball",
            (ball_x + 10, ball_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.ball_color,
            2,
        )

        return frame

    def _draw_rally_state(
        self, frame: np.ndarray, rally_state: Optional[str]
    ) -> np.ndarray:
        """Draw rally state on frame."""
        if rally_state is None:
            return frame

        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Draw rally state text
        state_text = f"Rally State: {rally_state}"
        cv2.putText(
            frame,
            state_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame

    def _draw_stroke_types(
        self, frame: np.ndarray, player1: PlayerData, player2: PlayerData
    ) -> np.ndarray:
        """Draw stroke types for both players."""
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 60), (350, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Draw Player 1 stroke type
        p1_color = self.player_colors.get(1, (255, 255, 255))
        p1_text = f"P1 Stroke: {player1.stroke_type}"
        cv2.putText(
            frame,
            p1_text,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            p1_color,
            2,
        )

        # Draw Player 2 stroke type
        p2_color = self.player_colors.get(2, (255, 255, 255))
        p2_text = f"P2 Stroke: {player2.stroke_type}"
        cv2.putText(
            frame,
            p2_text,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            p2_color,
            2,
        )

        return frame
