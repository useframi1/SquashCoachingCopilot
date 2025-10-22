"""Visualization module for rendering pipeline results on frames."""

import cv2
import numpy as np
import supervision as sv
from typing import Optional, Dict, List, Generator, Tuple
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
        show_wall_hits: bool = True,
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
            show_wall_hits: Whether to display wall hit indicators
            keypoint_confidence_threshold: Minimum confidence for displaying keypoints
        """
        self.show_court_keypoints = show_court_keypoints
        self.show_player_keypoints = show_player_keypoints
        self.show_player_bbox = show_player_bbox
        self.show_ball = show_ball
        self.show_rally_state = show_rally_state
        self.show_stroke_type = show_stroke_type
        self.show_wall_hits = show_wall_hits
        self.keypoint_confidence_threshold = keypoint_confidence_threshold

        # Professional color scheme: dark red, black, grey (BGR format)
        self.player_colors = {
            1: (25, 25, 139),  # Dark red for Player 1
            2: (100, 100, 100),  # Grey for Player 2
        }
        self.ellipse_colors = {
            1: (25, 25, 139),  # Dark red ellipse for Player 1
            2: (100, 100, 100),  # Grey ellipse for Player 2
        }
        self.ball_color = (25, 25, 139)  # Dark red for ball
        self.court_color = (100, 100, 100)  # Grey for court
        self.wall_hit_color = (25, 25, 139)  # Dark red for wall hits
        self.text_color = (200, 200, 200)  # Light grey for text
        self.bg_color = (20, 20, 20)  # Near black for backgrounds

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
        """Draw court keypoints on frame with professional styling."""
        if keypoints is None:
            return frame

        for class_name, kp_array in keypoints.items():
            for i, (x, y) in enumerate(kp_array):
                # Draw small circles with anti-aliasing
                cv2.circle(frame, (int(x), int(y)), 2, self.court_color, -1, cv2.LINE_AA)
                # Add subtle ring around keypoint
                cv2.circle(frame, (int(x), int(y)), 4, self.court_color, 1, cv2.LINE_AA)

        return frame

    def _draw_player(self, frame: np.ndarray, player_data: PlayerData) -> np.ndarray:
        """Draw player tracking results with ellipse under feet and professional styling."""
        if not player_data.is_valid():
            return frame

        player_id = player_data.player_id
        color = self.player_colors.get(player_id, (100, 100, 100))
        ellipse_color = self.ellipse_colors.get(player_id, (100, 100, 100))

        x1, y1, x2, y2 = map(int, player_data.bbox)
        bbox_width = x2 - x1

        # Draw ellipse under player's feet (at bottom-center of bbox)
        pos = player_data.position
        center_x, center_y = int(pos[0]), int(pos[1])

        # Ellipse dimensions based on bbox width
        ellipse_width = int(bbox_width * 0.6)  # 60% of bbox width
        ellipse_height = int(ellipse_width * 0.4)  # Make it flatter

        # Draw filled ellipse with slight transparency effect
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (ellipse_width // 2, ellipse_height // 2),
            0,  # rotation angle
            0,  # start angle
            360,  # end angle
            ellipse_color,
            -1  # filled
        )
        # Blend with original frame for semi-transparency
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw ellipse border for definition
        cv2.ellipse(
            frame,
            (center_x, center_y),
            (ellipse_width // 2, ellipse_height // 2),
            0,
            0,
            360,
            ellipse_color,
            2
        )

        # Draw minimal bounding box (thin, professional)
        if self.show_player_bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Draw player label above head (top of bbox)
        label = f"Player {player_id}"

        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Position text above the bounding box
        text_x = x1
        text_y = y1 - 10
        padding = 4

        # Draw semi-transparent background for text
        cv2.rectangle(
            frame,
            (text_x - padding, text_y - text_height - padding),
            (text_x + text_width + padding, text_y + baseline + padding),
            self.bg_color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            font,
            font_scale,
            self.text_color,
            thickness,
            cv2.LINE_AA
        )

        # Draw keypoints and skeleton
        if self.show_player_keypoints and player_data.keypoints:
            frame = self._draw_player_keypoints(frame, player_data, color)

        return frame

    def _draw_player_keypoints(
        self, frame: np.ndarray, player_data: PlayerData, color: tuple
    ) -> np.ndarray:
        """Draw player pose keypoints and skeleton with professional styling."""
        keypoints = player_data.keypoints
        if not keypoints or not keypoints.get("xy"):
            return frame

        kp_array = keypoints["xy"]
        conf_array = keypoints.get("conf")

        # Draw skeleton lines first (so they appear behind keypoints) - thinner, more subtle
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
                        # Draw with anti-aliasing for smoother lines
                        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw keypoints on top - smaller, more professional
        for i, (kp_x, kp_y) in enumerate(kp_array):
            if conf_array and conf_array[i] > self.keypoint_confidence_threshold:
                cv2.circle(frame, (int(kp_x), int(kp_y)), 2, color, -1, cv2.LINE_AA)

        return frame

    def _draw_ball(self, frame: np.ndarray, ball_data: BallData) -> np.ndarray:
        """Draw ball detection results with professional styling."""
        if not ball_data.is_valid():
            return frame

        ball_x, ball_y = ball_data.position

        # If ball is hitting wall, draw with emphasis
        if self.show_wall_hits and ball_data.is_wall_hit:
            # Draw outer glow effect for wall hit with transparency
            overlay = frame.copy()
            cv2.circle(overlay, (ball_x, ball_y), 15, self.wall_hit_color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Draw concentric circles for emphasis
            cv2.circle(frame, (ball_x, ball_y), 12, self.wall_hit_color, 1, cv2.LINE_AA)
            cv2.circle(frame, (ball_x, ball_y), 8, self.wall_hit_color, 1, cv2.LINE_AA)

            # Draw filled inner circle
            cv2.circle(frame, (ball_x, ball_y), 4, self.wall_hit_color, -1, cv2.LINE_AA)

            # Draw wall hit text with background
            label = "WALL HIT"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            text_x = ball_x + 15
            text_y = ball_y - 10
            padding = 3
            cv2.rectangle(
                frame,
                (text_x - padding, text_y - text_height - padding),
                (text_x + text_width + padding, text_y + baseline + padding),
                self.bg_color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                font,
                font_scale,
                self.wall_hit_color,
                thickness,
                cv2.LINE_AA
            )
        else:
            # Normal ball visualization - simple and clean
            cv2.circle(frame, (ball_x, ball_y), 4, self.ball_color, -1, cv2.LINE_AA)
            # Optional outer ring for visibility
            cv2.circle(frame, (ball_x, ball_y), 6, self.ball_color, 1, cv2.LINE_AA)

        return frame

    def _draw_rally_state(
        self, frame: np.ndarray, rally_state: Optional[str]
    ) -> np.ndarray:
        """Draw rally state on frame with professional styling."""
        if rally_state is None:
            return frame

        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 50), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Draw border
        cv2.rectangle(frame, (10, 10), (300, 50), self.text_color, 1, cv2.LINE_AA)

        # Draw rally state text
        state_text = f"Rally State: {rally_state}"
        cv2.putText(
            frame,
            state_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.text_color,
            1,
            cv2.LINE_AA
        )

        return frame

    def _draw_stroke_types(
        self, frame: np.ndarray, player1: PlayerData, player2: PlayerData
    ) -> np.ndarray:
        """Draw stroke types for both players with professional styling."""
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 60), (350, 130), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Draw border
        cv2.rectangle(frame, (10, 60), (350, 130), self.text_color, 1, cv2.LINE_AA)

        # Draw Player 1 stroke type with color indicator
        p1_color = self.player_colors.get(1, (100, 100, 100))
        p1_text = f"P1 Stroke: {player1.stroke_type}"

        # Draw color indicator circle
        cv2.circle(frame, (25, 80), 6, p1_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (25, 80), 6, self.text_color, 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            p1_text,
            (40, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
            cv2.LINE_AA
        )

        # Draw Player 2 stroke type with color indicator
        p2_color = self.player_colors.get(2, (100, 100, 100))
        p2_text = f"P2 Stroke: {player2.stroke_type}"

        # Draw color indicator circle
        cv2.circle(frame, (25, 110), 6, p2_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (25, 110), 6, self.text_color, 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            p2_text,
            (40, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
            cv2.LINE_AA
        )

        return frame

    def render_frames(
        self,
        frames: Generator[Tuple[int, float, np.ndarray], None, None],
        frame_data_list: List[FrameData],
    ) -> Generator[np.ndarray, None, None]:
        """
        Render multiple frames with post-processed data.

        Args:
            frames: Generator yielding (frame_number, timestamp, frame) tuples
            frame_data_list: List of post-processed FrameData objects

        Yields:
            Annotated frames
        """
        print("\nRendering post-processed frames...")

        # Create a mapping from frame_number to frame_data for fast lookup
        frame_data_map = {fd.frame_number: fd for fd in frame_data_list}

        for frame_number, _, frame in frames:
            # Get the post-processed frame data for this frame number
            if frame_number in frame_data_map:
                frame_data = frame_data_map[frame_number]

                # Render frame with post-processed data
                annotated_frame = self.render_frame(frame, frame_data)

                yield annotated_frame
