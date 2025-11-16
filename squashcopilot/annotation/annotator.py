"""
Video annotation module for squash coaching analysis.

This module provides functionality to annotate squash videos with:
- Player tracking (positions and keypoints)
- Ball tracking
- Wall hit detection
- Racket hit detection
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from squashcopilot.modules.court_calibration import CourtCalibrator
from squashcopilot.modules.player_tracking import PlayerTracker
from squashcopilot.modules.ball_tracking import (
    BallTracker,
    WallHitDetector,
    RacketHitDetector,
)
from squashcopilot.common import (
    Frame,
    CourtCalibrationInput,
    WallColorDetectionInput,
    PlayerTrackingInput,
    BallTrackingInput,
    PlayerPostprocessingInput,
    BallPostprocessingInput,
    WallHitInput,
    RacketHitInput,
    Point2D,
    Homography,
    BODY_KEYPOINT_INDICES,
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
)
from squashcopilot.common.utils import load_config


class Annotator:
    """
    Main annotator class for processing squash videos.

    This class handles the complete pipeline:
    1. Court calibration
    2. Player and ball tracking
    3. Hit detection
    4. CSV export with annotations
    5. Annotated video generation
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the annotator with all required modules."""
        # Load configuration
        if config is None:
            self.config = load_config(config_name='annotation')
        else:
            self.config = config

        # Get the annotation module directory
        self.module_dir = Path(__file__).parent
        self.data_dir = self.module_dir / "data"
        self.annotations_dir = self.module_dir / "annotations"

        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)

        # Initialize processing modules
        self.court_calibrator = CourtCalibrator()
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.wall_hit_detector = WallHitDetector()
        self.racket_hit_detector = RacketHitDetector()

    def annotate_video(self, video_name: str) -> Tuple[str, str]:
        """
        Process a video and generate annotations.

        Args:
            video_name: Name of the video file in the data directory

        Returns:
            Tuple of (csv_path, video_path) for the generated annotations

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If court calibration fails
        """
        video_path = self.data_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        max_frames = self.config.get('video', {}).get('max_frames')

        print(f"Processing video: {video_name}")

        # Step 1: Load video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Limit frames if max_frames is specified
        if max_frames is not None and max_frames > 0:
            total_frames = min(total_frames, max_frames)
            print(f"Video info: {total_frames} frames (limited from {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}), {fps:.2f} FPS, {width}x{height}")
        else:
            print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")

        # Step 2: Calibrate court on first frame
        ret, first_frame_img = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")

        first_frame = Frame(image=first_frame_img, frame_number=0, timestamp=0.0)

        print("Calibrating court...")
        court_result = self.court_calibrator.process_frame(
            CourtCalibrationInput(frame=first_frame)
        )

        if not court_result.calibrated:
            raise RuntimeError("Court calibration failed on first frame")

        floor_homography = court_result.get_homography('floor')
        court_keypoints = court_result.keypoints_per_class

        print("Court calibration successful")

        # Step 3: Detect wall color and set ball color
        print("Detecting wall color...")
        wall_color_result = self.court_calibrator.detect_wall_color(
            WallColorDetectionInput(
                frame=first_frame,
                keypoints_per_class=court_keypoints
            )
        )

        is_black_ball = wall_color_result.is_white
        self.ball_tracker.set_is_black_ball(is_black_ball)
        print(f"Wall is {'white' if wall_color_result.is_white else 'colored'}, using {'black' if is_black_ball else 'white'} ball")

        # Step 4: Update player tracker with homography
        self.player_tracker.homography = floor_homography.matrix

        # Step 5: Process all frames
        print("Tracking players and ball...")
        ball_positions_raw = []
        player_positions_raw = {1: [], 2: []}
        player_real_positions_raw = {1: [], 2: []}
        player_keypoints_raw = {1: [], 2: []}
        player_bboxes_raw = {1: [], 2: []}

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

        for frame_idx in tqdm(range(total_frames), desc="Tracking", unit="frame"):
            ret, frame_img = cap.read()
            if not ret:
                break

            frame = Frame(
                image=frame_img,
                frame_number=frame_idx,
                timestamp=frame_idx / fps
            )

            # Track players
            player_result = self.player_tracker.process_frame(
                PlayerTrackingInput(frame=frame, homography=floor_homography)
            )

            for player_id in [1, 2]:
                player = player_result.get_player(player_id)
                if player:
                    player_positions_raw[player_id].append(player.position)
                    player_real_positions_raw[player_id].append(player.real_position)
                    player_keypoints_raw[player_id].append(player.keypoints)
                    player_bboxes_raw[player_id].append(player.bbox)
                else:
                    player_positions_raw[player_id].append(None)
                    player_real_positions_raw[player_id].append(None)
                    player_keypoints_raw[player_id].append(None)
                    player_bboxes_raw[player_id].append(None)

            # Track ball
            ball_result = self.ball_tracker.process_frame(
                BallTrackingInput(frame=frame)
            )
            ball_positions_raw.append(
                ball_result.position if ball_result.detected else None
            )

        cap.release()

        # Step 6: Postprocess player trajectories
        print("Postprocessing player trajectories...")
        player_postprocess_result = self.player_tracker.postprocess(
            PlayerPostprocessingInput(
                positions_history=player_positions_raw,
                real_positions_history=player_real_positions_raw,
                keypoints_history=player_keypoints_raw,
                bboxes_history=player_bboxes_raw
            )
        )

        player_positions_smooth = {
            player_id: traj.positions
            for player_id, traj in player_postprocess_result.trajectories.items()
        }

        # Extract interpolated keypoints and bboxes
        player_keypoints_interpolated = {
            player_id: traj.keypoints if traj.keypoints else player_keypoints_raw[player_id]
            for player_id, traj in player_postprocess_result.trajectories.items()
        }

        player_bboxes_interpolated = {
            player_id: traj.bboxes if traj.bboxes else player_bboxes_raw[player_id]
            for player_id, traj in player_postprocess_result.trajectories.items()
        }

        for player_id, traj in player_postprocess_result.trajectories.items():
            print(f"Player {player_id}: {traj.gaps_filled} gaps filled")

        # Step 7: Postprocess ball trajectory
        print("Postprocessing ball trajectory...")
        ball_trajectory = self.ball_tracker.postprocess(
            BallPostprocessingInput(positions=ball_positions_raw)
        )
        ball_positions_smooth = ball_trajectory.positions
        print(f"Ball: {ball_trajectory.outliers_removed} outliers removed, {ball_trajectory.gaps_filled} gaps filled")

        # Step 8: Detect wall hits
        print("Detecting wall hits...")
        wall_hits_result = self.wall_hit_detector.detect(
            WallHitInput(positions=ball_positions_smooth)
        )
        print(f"Detected {wall_hits_result.num_hits} wall hits")

        # Step 9: Detect racket hits
        print("Detecting racket hits...")
        racket_hits_result = self.racket_hit_detector.detect(
            RacketHitInput(
                positions=ball_positions_smooth,
                wall_hits=wall_hits_result.wall_hits
            )
        )
        print(f"Detected {racket_hits_result.num_hits} racket hits")

        # Step 10: Build DataFrame
        print("Building DataFrame...")
        df = self._build_dataframe(
            total_frames=total_frames,
            player_positions_smooth=player_positions_smooth,
            player_keypoints_raw=player_keypoints_interpolated,
            ball_positions_smooth=ball_positions_smooth,
            wall_hits=wall_hits_result.wall_hits,
            racket_hits=racket_hits_result.racket_hits,
            floor_homography=floor_homography
        )

        # Step 11: Save CSV
        video_base_name = Path(video_name).stem
        csv_path = self.annotations_dir / f"{video_base_name}_annotations.csv"
        csv_index = self.config.get('output', {}).get('csv_index', False)
        df.to_csv(csv_path, index=csv_index)
        print(f"Saved annotations to: {csv_path}")

        # Step 12: Create annotated video
        print("Creating annotated video...")
        annotated_video_path = self.annotations_dir / f"{video_base_name}_annotated.mp4"
        self._create_annotated_video(
            video_path=video_path,
            output_path=annotated_video_path,
            court_keypoints=court_keypoints,
            player_positions=player_positions_smooth,
            player_keypoints=player_keypoints_interpolated,
            player_bboxes=player_bboxes_interpolated,
            ball_positions=ball_positions_smooth,
            wall_hits=wall_hits_result.wall_hits,
            racket_hits=racket_hits_result.racket_hits,
            fps=fps,
            num_frames=total_frames
        )
        print(f"Saved annotated video to: {annotated_video_path}")

        return str(csv_path), str(annotated_video_path)

    def _build_dataframe(
        self,
        total_frames: int,
        player_positions_smooth: Dict[int, List[Point2D]],
        player_keypoints_raw: Dict[int, List],
        ball_positions_smooth: List[Point2D],
        wall_hits: List,
        racket_hits: List,
        floor_homography: Homography
    ) -> pd.DataFrame:
        """Build the annotations DataFrame."""
        # Create wall and racket hit frame sets for quick lookup
        wall_hit_frames = {hit.frame for hit in wall_hits}
        racket_hit_frames = {hit.frame for hit in racket_hits}

        # Build rows
        rows = []
        for frame_idx in range(total_frames):
            row = {'frame': frame_idx}

            # Add player data
            for player_id in [1, 2]:
                prefix = f'player_{player_id}'

                # Positions (pixel)
                pos_pixel = player_positions_smooth[player_id][frame_idx]
                row[f'{prefix}_x_pixel'] = pos_pixel.x
                row[f'{prefix}_y_pixel'] = pos_pixel.y

                # Positions (meter) - transform using homography
                pos_meter = floor_homography.transform_point(pos_pixel)
                row[f'{prefix}_x_meter'] = pos_meter.x
                row[f'{prefix}_y_meter'] = pos_meter.y

                # Keypoints (extract body keypoints only)
                kp_data = player_keypoints_raw[player_id][frame_idx]
                if kp_data and kp_data.xy:
                    # Extract COCO keypoints 5-16 (12 body keypoints)
                    for i, kp_idx in enumerate(BODY_KEYPOINT_INDICES):
                        kp_name = KEYPOINT_NAMES[i]
                        kp_result = kp_data.get_keypoint(kp_idx)
                        if kp_result is not None:
                            x, y, conf = kp_result
                            row[f'{prefix}_kp_{kp_name}_x'] = x if conf > 0 else None
                            row[f'{prefix}_kp_{kp_name}_y'] = y if conf > 0 else None
                        else:
                            row[f'{prefix}_kp_{kp_name}_x'] = None
                            row[f'{prefix}_kp_{kp_name}_y'] = None
                else:
                    # No keypoints detected
                    for kp_name in KEYPOINT_NAMES:
                        row[f'{prefix}_kp_{kp_name}_x'] = None
                        row[f'{prefix}_kp_{kp_name}_y'] = None

            # Ball position
            ball_pos = ball_positions_smooth[frame_idx]
            row['ball_x'] = ball_pos.x
            row['ball_y'] = ball_pos.y

            # Hit flags
            row['is_wall_hit'] = frame_idx in wall_hit_frames
            row['is_racket_hit'] = frame_idx in racket_hit_frames

            rows.append(row)

        return pd.DataFrame(rows)

    def _create_annotated_video(
        self,
        video_path: Path,
        output_path: Path,
        court_keypoints: Dict,
        player_positions: Dict[int, List[Point2D]],
        player_keypoints: Dict[int, List],
        player_bboxes: Dict[int, List],
        ball_positions: List[Point2D],
        wall_hits: List,
        racket_hits: List,
        fps: float,
        num_frames: int
    ):
        """Create an annotated video with visualizations."""
        # Load visualization config
        viz_config = self.config.get('visualization', {})
        ball_trail_length = viz_config.get('ball_trail_length', 30)

        # Get colors (convert from list to tuple for OpenCV)
        player_colors = {
            1: tuple(viz_config.get('player_colors', {}).get('player_1', [0, 255, 0])),
            2: tuple(viz_config.get('player_colors', {}).get('player_2', [255, 0, 0]))
        }
        ball_color = tuple(viz_config.get('ball_color', [0, 255, 255]))
        court_color = tuple(viz_config.get('court_color', [255, 255, 0]))
        wall_hit_color = tuple(viz_config.get('wall_hit_color', [0, 0, 255]))
        racket_hit_color = tuple(viz_config.get('racket_hit_color', [255, 0, 255]))

        # Create dictionaries mapping frame to hit position
        wall_hit_positions = {hit.frame: hit.position for hit in wall_hits}
        racket_hit_positions = {hit.frame: hit.position for hit in racket_hits}

        # Calculate hit marker duration in frames (1 second)
        hit_marker_duration = int(fps * 1)

        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        codec = self.config.get('output', {}).get('video_codec', 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Get drawing parameters from config
        bbox_thickness = viz_config.get('bbox_thickness', 2)
        label_font_scale = viz_config.get('label_font_scale', 0.7)
        label_font_thickness = viz_config.get('label_font_thickness', 2)
        wall_hit_config = viz_config.get('wall_hit_marker', {})
        racket_hit_config = viz_config.get('racket_hit_marker', {})

        for frame_idx in tqdm(range(num_frames), desc="Creating video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw court calibration overlay
            self._draw_court_overlay(frame, court_keypoints, court_color)

            # Draw players
            for player_id in [1, 2]:
                bbox = player_bboxes[player_id][frame_idx]
                kp_data = player_keypoints[player_id][frame_idx]
                pos = player_positions[player_id][frame_idx]

                if bbox and kp_data:
                    # Draw bounding box
                    color = player_colors[player_id]
                    cv2.rectangle(
                        frame,
                        (int(bbox.x1), int(bbox.y1)),
                        (int(bbox.x2), int(bbox.y2)),
                        color,
                        bbox_thickness
                    )

                    # Draw player ID
                    cv2.putText(
                        frame,
                        f'P{player_id}',
                        (int(bbox.x1), int(bbox.y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale,
                        color,
                        label_font_thickness
                    )

                    # Draw keypoints and skeleton
                    self._draw_keypoints(frame, kp_data, color, viz_config)

            # Draw ball position
            ball_pos = ball_positions[frame_idx]
            cv2.circle(frame, (int(ball_pos.x), int(ball_pos.y)), 5, ball_color, -1)

            # Draw ball trajectory (trail)
            start_idx = max(0, frame_idx - ball_trail_length)
            for i in range(start_idx, frame_idx):
                p1 = ball_positions[i]
                p2 = ball_positions[i + 1]
                # Fade effect
                alpha = (i - start_idx) / ball_trail_length if ball_trail_length > 0 else 1
                # Create faded version of ball_color
                faded_color = tuple(int(c * alpha) for c in ball_color)
                cv2.line(
                    frame,
                    (int(p1.x), int(p1.y)),
                    (int(p2.x), int(p2.y)),
                    faded_color,
                    2
                )

            # Draw hit markers (persist for 1 second)
            wall_text = wall_hit_config.get('text', 'WALL HIT')
            wall_radius = wall_hit_config.get('circle_radius', 20)

            racket_text = racket_hit_config.get('text', 'RACKET HIT')
            racket_radius = racket_hit_config.get('circle_radius', 25)

            # Small text size for hit labels
            hit_text_scale = 0.5
            hit_text_thickness = 1

            # Draw wall hit markers (persist for hit_marker_duration frames)
            for hit_frame, hit_pos in wall_hit_positions.items():
                if hit_frame <= frame_idx < hit_frame + hit_marker_duration:
                    # Draw marker at hit position
                    cv2.circle(frame, (int(hit_pos.x), int(hit_pos.y)), wall_radius, wall_hit_color, 3)
                    # Draw X marker at hit position for better visibility
                    marker_size = 15
                    cv2.line(
                        frame,
                        (int(hit_pos.x) - marker_size, int(hit_pos.y) - marker_size),
                        (int(hit_pos.x) + marker_size, int(hit_pos.y) + marker_size),
                        wall_hit_color,
                        3
                    )
                    cv2.line(
                        frame,
                        (int(hit_pos.x) - marker_size, int(hit_pos.y) + marker_size),
                        (int(hit_pos.x) + marker_size, int(hit_pos.y) - marker_size),
                        wall_hit_color,
                        3
                    )
                    # Draw small text label above the marker
                    text_offset_y = 30  # Pixels above the marker
                    cv2.putText(
                        frame,
                        wall_text,
                        (int(hit_pos.x) - 30, int(hit_pos.y) - text_offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        hit_text_scale,
                        wall_hit_color,
                        hit_text_thickness
                    )

            # Draw racket hit markers (persist for hit_marker_duration frames)
            for hit_frame, hit_pos in racket_hit_positions.items():
                if hit_frame <= frame_idx < hit_frame + hit_marker_duration:
                    # Draw marker at hit position
                    cv2.circle(frame, (int(hit_pos.x), int(hit_pos.y)), racket_radius, racket_hit_color, 3)
                    # Draw X marker at hit position for better visibility
                    marker_size = 15
                    cv2.line(
                        frame,
                        (int(hit_pos.x) - marker_size, int(hit_pos.y) - marker_size),
                        (int(hit_pos.x) + marker_size, int(hit_pos.y) + marker_size),
                        racket_hit_color,
                        3
                    )
                    cv2.line(
                        frame,
                        (int(hit_pos.x) - marker_size, int(hit_pos.y) + marker_size),
                        (int(hit_pos.x) + marker_size, int(hit_pos.y) - marker_size),
                        racket_hit_color,
                        3
                    )
                    # Draw small text label above the marker
                    text_offset_y = 30  # Pixels above the marker
                    cv2.putText(
                        frame,
                        racket_text,
                        (int(hit_pos.x) - 40, int(hit_pos.y) - text_offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        hit_text_scale,
                        racket_hit_color,
                        hit_text_thickness
                    )

            out.write(frame)

        cap.release()
        out.release()

    def _draw_court_overlay(self, frame: np.ndarray, court_keypoints: Dict, color: tuple):
        """Draw court calibration overlay."""
        viz_config = self.config.get('visualization', {})
        court_thickness = viz_config.get('court_thickness', 2)

        # Draw keypoints for each court element
        for class_name, keypoints in court_keypoints.items():
            # Get the 4 corners
            points = [
                keypoints.get_point('top_left'),
                keypoints.get_point('top_right'),
                keypoints.get_point('bottom_right'),
                keypoints.get_point('bottom_left')
            ]

            # Draw lines connecting the corners
            pts = np.array([[int(p.x), int(p.y)] for p in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, court_thickness)

            # Draw corner points
            for point in points:
                cv2.circle(frame, (int(point.x), int(point.y)), 3, color, -1)

    def _draw_keypoints(self, frame: np.ndarray, keypoints_data, color: tuple, viz_config: Dict):
        """Draw player keypoints and skeleton."""
        if not keypoints_data or not keypoints_data.xy:
            return

        skeleton_thickness = viz_config.get('skeleton_thickness', 2)
        keypoint_radius = viz_config.get('keypoint_radius', 3)
        keypoint_conf_threshold = viz_config.get('keypoint_confidence_threshold', 0.5)

        # Draw skeleton connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            kp1 = keypoints_data.get_keypoint(start_idx)
            kp2 = keypoints_data.get_keypoint(end_idx)

            if kp1 is not None and kp2 is not None:
                x1, y1, conf1 = kp1
                x2, y2, conf2 = kp2

                # Ensure x and y are scalar values (not lists)
                if isinstance(x1, (list, tuple)):
                    x1 = x1[0] if len(x1) > 0 else 0
                if isinstance(y1, (list, tuple)):
                    y1 = y1[0] if len(y1) > 0 else 0
                if isinstance(x2, (list, tuple)):
                    x2 = x2[0] if len(x2) > 0 else 0
                if isinstance(y2, (list, tuple)):
                    y2 = y2[0] if len(y2) > 0 else 0

                if conf1 > keypoint_conf_threshold and conf2 > keypoint_conf_threshold:
                    cv2.line(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        skeleton_thickness
                    )

        # Draw keypoint circles
        for idx in BODY_KEYPOINT_INDICES:
            kp = keypoints_data.get_keypoint(idx)
            if kp is not None:
                x, y, conf = kp

                # Ensure x and y are scalar values (not lists)
                if isinstance(x, (list, tuple)):
                    x = x[0] if len(x) > 0 else 0
                if isinstance(y, (list, tuple)):
                    y = y[0] if len(y) > 0 else 0

                if conf > keypoint_conf_threshold:
                    cv2.circle(frame, (int(x), int(y)), keypoint_radius, color, -1)

    def load_annotations(self, video_name: str) -> pd.DataFrame:
        """
        Load annotations from CSV file.

        Args:
            video_name: Name of the video (without or with extension)

        Returns:
            DataFrame with annotations

        Raises:
            FileNotFoundError: If annotations file doesn't exist
        """
        video_base_name = Path(video_name).stem
        csv_path = self.annotations_dir / f"{video_base_name}_annotations.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotations not found: {csv_path}")

        return pd.read_csv(csv_path)


if __name__ == "__main__":
    import sys

    # Initialize annotator
    annotator = Annotator()

    # Get video name from config
    video_name = annotator.config.get('video', {}).get('name')
    if not video_name:
        print("Error: No video name specified in config", file=sys.stderr)
        sys.exit(1)

    try:
        # Process video
        print(f"Starting annotation process for: {video_name}\n")
        csv_path, video_path = annotator.annotate_video(video_name)
        print(f"\n{'='*60}")
        print(f"Annotation complete!")
        print(f"{'='*60}")
        print(f"CSV saved to: {csv_path}")
        print(f"Video saved to: {video_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
