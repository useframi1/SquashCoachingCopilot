"""
Video annotation module for squash coaching analysis.

This module provides the Annotator class that coordinates all stages of video processing
from raw input video to comprehensive analysis outputs (CSV with keypoints, annotated video).

Uses DataFrame-based data flow architecture identical to pipeline.py.
The key difference from Pipeline is that keypoints are saved to the CSV output.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from squashcopilot.common.types import Frame
from squashcopilot.common.utils import load_config
from squashcopilot.common.constants import KEYPOINT_NAMES, SKELETON_CONNECTIONS
from squashcopilot.common.models import (
    VideoMetadata,
    CourtCalibrationInput,
    CourtCalibrationOutput,
    PlayerTrackingInput,
    PlayerTrackingOutput,
    player_tracking_outputs_to_dataframe,
    PlayerPostprocessingInput,
    PlayerPostprocessingOutput,
    BallTrackingInput,
    BallTrackingOutput,
    ball_tracking_outputs_to_dataframe,
    BallPostprocessingInput,
    BallPostprocessingOutput,
    RallySegmentationInput,
    RallySegmentationOutput,
    RallySegment,
    WallHitDetectionInput,
    WallHitDetectionOutput,
    RacketHitDetectionInput,
    RacketHitDetectionOutput,
    StrokeClassificationInput,
    StrokeClassificationOutput,
    ShotClassificationInput,
    ShotClassificationOutput,
    PipelineSession,
)

from squashcopilot.modules.court_calibration.court_calibrator import CourtCalibrator
from squashcopilot.modules.player_tracking.player_tracker import PlayerTracker
from squashcopilot.modules.ball_tracking.ball_tracker import BallTracker
from squashcopilot.modules.hit_detection.wall_hit_detector import WallHitDetector
from squashcopilot.modules.hit_detection.racket_hit_detector import RacketHitDetector
from squashcopilot.modules.rally_state_detection.rally_state_detector import (
    RallyStateDetector,
)
from squashcopilot.modules.stroke_detection.stroke_detector import StrokeDetector
from squashcopilot.modules.shot_type_classification.shot_classifier import (
    ShotClassifier,
)


class Annotator:
    """
    Complete annotator for squash video analysis.

    Coordinates all 7 stages of processing with DataFrame-based data flow:
    1. Court calibration (first frame)
    2. Frame-by-frame tracking (player + ball) -> DataFrame
    3. Trajectory postprocessing (smoothing, interpolation)
    4. Rally segmentation (identify rally boundaries)
    5. Hit detection (wall hits + racket hits)
    6. Stroke and shot classification
    7. Export (CSV with keypoints, annotated video)

    The key difference from Pipeline is that keypoints are saved to the CSV output.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the annotator with configuration.

        Args:
            config_path: Path to custom annotation config YAML. If None, uses default config.
        """
        if config_path:
            self.config = load_config(config_path=config_path)
        else:
            self.config = load_config(config_name="annotation")

        self._setup_logging()
        self.logger.info("Initializing annotator modules...")
        self._initialize_modules()
        self._setup_paths()
        self.logger.info("Annotator initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_modules(self):
        """Initialize all processing modules."""
        self.court_calibrator = CourtCalibrator()
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.wall_hit_detector = WallHitDetector()
        self.racket_hit_detector = RacketHitDetector()
        self.rally_detector = RallyStateDetector()
        self.stroke_detector = StrokeDetector()
        self.shot_classifier = ShotClassifier()

    def _setup_paths(self):
        """Setup directory paths from config."""
        # Get project root
        self.module_dir = Path(__file__).parent
        self.project_root = self.module_dir.parent.parent

        # Video directory from config
        video_dir_rel = self.config.get("video", {}).get(
            "video_dir", "squashcopilot/videos"
        )
        self.video_dir = self.project_root / video_dir_rel

        # Annotations output directory
        self.annotations_dir = self.module_dir / "annotations"
        self.annotations_dir.mkdir(exist_ok=True)

    def run(self, video_name: Optional[str] = None) -> Dict[str, str]:
        """
        Run the complete annotation pipeline on a video.

        Args:
            video_name: Name of video file (without extension). If None, uses config.

        Returns:
            Dictionary with paths to generated outputs
        """
        # Get video name from argument or config
        if video_name is None:
            video_name = self.config.get("video", {}).get("name")
        if not video_name:
            raise ValueError("video_name not specified")

        # Build video path
        video_base_name = Path(video_name).stem
        video_path = self.video_dir / f"{video_base_name}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Starting annotation for video: {video_path}")
        start_time = time.time()

        # Setup output directory
        output_dir = self.annotations_dir / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video metadata
        video_metadata = self._get_video_metadata(str(video_path))
        self.logger.info(
            f"Video info: {video_metadata.total_frames} frames, "
            f"{video_metadata.fps:.2f} FPS, {video_metadata.duration_seconds:.2f}s"
        )

        # Initialize session
        session = PipelineSession(
            video_metadata=video_metadata,
            calibration=None,
            current_df=None,
            complex_data={},
            rally_segments=[],
            processing_stats={},
        )

        # Stage 1: Court Calibration
        session.calibration = self._stage1_calibrate_court(str(video_path))

        # Stage 2: Frame-by-frame Tracking -> DataFrame
        session.current_df, session.complex_data = self._stage2_track_frames(
            str(video_path), session.calibration, video_metadata
        )

        # Stage 3: Trajectory Postprocessing
        session.current_df, session.complex_data = self._stage3_postprocess(
            session.current_df, session.complex_data
        )

        # Stage 4: Rally Segmentation
        session.current_df, session.rally_segments = self._stage4_segment_rallies(
            session.current_df, session.calibration
        )

        # Stage 5: Hit Detection
        session.current_df = self._stage5_detect_hits(
            session.current_df, session.rally_segments, session.calibration
        )

        # Stage 6: Stroke and Shot Classification
        session.current_df = self._stage6_classify(
            session.current_df,
            session.complex_data,
            session.rally_segments,
            session.calibration,
        )

        # Stage 7: Export Results (with keypoints in CSV)
        output_paths = self._stage7_export(
            video_path=str(video_path),
            video_name=video_base_name,
            output_dir=output_dir,
            session=session,
        )

        total_time = time.time() - start_time
        self.logger.info(f"Annotation completed in {total_time:.2f}s")

        return output_paths

    def _get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps == 0:
            raise ValueError(f"Video FPS is 0: {video_path}")
        if total_frames == 0:
            raise ValueError(f"Video has 0 frames: {video_path}")

        # Apply max_time limit if specified
        max_time = self.config.get("video", {}).get("max_time")
        if max_time is not None:
            max_frames = int(max_time * fps)
            if max_frames < total_frames:
                self.logger.info(f"Limiting to {max_time}s ({max_frames} frames)")
                total_frames = max_frames

        return VideoMetadata(
            filepath=Path(video_path),
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_seconds=total_frames / fps,
        )

    def _stage1_calibrate_court(self, video_path: str) -> CourtCalibrationOutput:
        """Stage 1: Court calibration from first frame."""
        self.logger.info("Stage 1: Court Calibration")
        stage_start = time.time()

        cap = cv2.VideoCapture(video_path)
        ret, frame_img = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read first frame from {video_path}")

        frame = Frame(image=frame_img, frame_number=0, timestamp=0.0)
        calib_input = CourtCalibrationInput(frame=frame)
        calibration = self.court_calibrator.process_frame(calib_input)

        if not calibration.calibration_success:
            raise RuntimeError("Court calibration failed on first frame")

        self.logger.info(f"Stage 1 completed in {time.time() - stage_start:.2f}s")
        return calibration

    def _stage2_track_frames(
        self,
        video_path: str,
        calibration: CourtCalibrationOutput,
        video_metadata: VideoMetadata,
    ) -> tuple:
        """Stage 2: Frame-by-frame tracking -> DataFrame."""
        self.logger.info("Stage 2: Frame-by-Frame Tracking")
        stage_start = time.time()

        cap = cv2.VideoCapture(video_path)
        total_frames = video_metadata.total_frames
        fps = video_metadata.fps

        player_outputs = []
        ball_outputs = []

        # Set ball tracker for black ball if needed
        if calibration:
            self.ball_tracker.set_is_black_ball(calibration.is_black_ball)

        # Set player tracker calibration
        if calibration:
            self.player_tracker.set_calibration(calibration)

        frame_number = 0
        pbar = tqdm(total=total_frames, desc="Tracking frames")

        while cap.isOpened() and frame_number < total_frames:
            ret, frame_img = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps
            frame = Frame(
                image=frame_img, frame_number=frame_number, timestamp=timestamp
            )

            # Track players
            player_tracking_input = PlayerTrackingInput(
                frame=frame, calibration=calibration
            )
            player_output = self.player_tracker.process_frame(player_tracking_input)

            # Track ball
            ball_tracking_input = BallTrackingInput(frame=frame)
            ball_output = self.ball_tracker.process_frame(ball_tracking_input)

            player_outputs.append(player_output)
            ball_outputs.append(ball_output)
            frame_number += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # Convert to DataFrames + complex data
        player_df, complex_data = player_tracking_outputs_to_dataframe(player_outputs)
        ball_df = ball_tracking_outputs_to_dataframe(ball_outputs)

        # Merge player and ball DataFrames
        df = player_df.join(ball_df.drop(columns=["timestamp"], errors="ignore"))

        self.logger.info(f"Stage 2 completed in {time.time() - stage_start:.2f}s")
        return df, complex_data

    def _stage3_postprocess(
        self,
        df: pd.DataFrame,
        complex_data: Dict,
    ) -> tuple:
        """Stage 3: Trajectory postprocessing using tracker postprocess methods."""
        self.logger.info("Stage 3: Trajectory Postprocessing")
        stage_start = time.time()

        # Build postprocessing input for players
        player_keypoints = {
            1: complex_data.get("player_1_keypoints", []),
            2: complex_data.get("player_2_keypoints", []),
        }
        player_bboxes = {
            1: complex_data.get("player_1_bboxes", []),
            2: complex_data.get("player_2_bboxes", []),
        }

        ball_outliers = 0
        ball_gaps = 0
        player_1_gaps = 0
        player_2_gaps = 0

        # Postprocess player trajectories
        player_postprocess_input = PlayerPostprocessingInput(
            df=df, player_keypoints=player_keypoints, player_bboxes=player_bboxes
        )
        player_postprocess_output: PlayerPostprocessingOutput = (
            self.player_tracker.postprocess(player_postprocess_input)
        )
        df = player_postprocess_output.df
        player_1_gaps = player_postprocess_output.num_player_1_gaps_filled
        player_2_gaps = player_postprocess_output.num_player_2_gaps_filled

        # Update complex_data with processed keypoints/bboxes
        complex_data["player_1_keypoints"] = (
            player_postprocess_output.player_keypoints.get(1, [])
        )
        complex_data["player_2_keypoints"] = (
            player_postprocess_output.player_keypoints.get(2, [])
        )
        complex_data["player_1_bboxes"] = player_postprocess_output.player_bboxes.get(
            1, []
        )
        complex_data["player_2_bboxes"] = player_postprocess_output.player_bboxes.get(
            2, []
        )

        # Postprocess ball trajectory
        ball_postprocess_input = BallPostprocessingInput(df=df)
        ball_postprocess_output: BallPostprocessingOutput = (
            self.ball_tracker.postprocess(ball_postprocess_input)
        )
        df = ball_postprocess_output.df
        ball_outliers = ball_postprocess_output.num_ball_outliers
        ball_gaps = ball_postprocess_output.num_ball_gaps_filled

        self.logger.info(
            f"Stage 3 completed in {time.time() - stage_start:.2f}s - "
            f"Player gaps: P1={player_1_gaps}, P2={player_2_gaps}, "
            f"Ball outliers: {ball_outliers}, gaps: {ball_gaps}"
        )
        return df, complex_data

    def _stage4_segment_rallies(
        self,
        df: pd.DataFrame,
        calibration: CourtCalibrationOutput,
    ) -> tuple:
        """Stage 4: Rally segmentation."""
        self.logger.info("Stage 4: Rally Segmentation")
        stage_start = time.time()

        rally_input = RallySegmentationInput(df=df, calibration=calibration)
        rally_output = self.rally_detector.segment_rallies(rally_input)

        self.logger.info(
            f"Stage 4 completed in {time.time() - stage_start:.2f}s - "
            f"Found {rally_output.num_rallies} rallies"
        )
        return rally_output.df, rally_output.segments

    def _stage5_detect_hits(
        self,
        df: pd.DataFrame,
        segments: List[RallySegment],
        calibration: CourtCalibrationOutput,
    ) -> pd.DataFrame:
        """Stage 5: Wall and racket hit detection."""
        self.logger.info("Stage 5: Hit Detection")
        stage_start = time.time()

        # Wall hit detection
        if segments:
            wall_hit_input = WallHitDetectionInput(
                df=df, segments=segments, calibration=calibration
            )
            wall_hit_output = self.wall_hit_detector.detect_wall_hits(wall_hit_input)
            df = wall_hit_output.df

        # Racket hit detection
        if segments and "is_wall_hit" in df.columns:
            racket_hit_input = RacketHitDetectionInput(df=df, segments=segments)
            racket_hit_output = self.racket_hit_detector.detect_racket_hits(
                racket_hit_input
            )
            df = racket_hit_output.df

        num_wall_hits = (
            int(df["is_wall_hit"].sum()) if "is_wall_hit" in df.columns else 0
        )
        num_racket_hits = (
            int(df["is_racket_hit"].sum()) if "is_racket_hit" in df.columns else 0
        )

        self.logger.info(
            f"Stage 5 completed in {time.time() - stage_start:.2f}s - "
            f"Wall hits: {num_wall_hits}, Racket hits: {num_racket_hits}"
        )
        return df

    def _stage6_classify(
        self,
        df: pd.DataFrame,
        complex_data: Dict,
        segments: List[RallySegment],
        calibration: CourtCalibrationOutput,
    ) -> pd.DataFrame:
        """Stage 6: Stroke and shot classification."""
        self.logger.info("Stage 6: Stroke and Shot Classification")
        stage_start = time.time()

        # Build player keypoints dict
        player_keypoints = {
            1: complex_data.get("player_1_keypoints", []),
            2: complex_data.get("player_2_keypoints", []),
        }

        # Stroke detection
        stroke_input = StrokeClassificationInput(
            df=df, player_keypoints=player_keypoints
        )
        stroke_output = self.stroke_detector.detect_strokes(stroke_input)
        df = stroke_output.df

        # Shot classification
        shot_input = ShotClassificationInput(df=df, segments=segments)
        shot_output = self.shot_classifier.classify_shots(shot_input)
        df = shot_output.df

        num_strokes = (
            int((df["stroke_type"] != "").sum()) if "stroke_type" in df.columns else 0
        )
        num_shots = (
            int((df["shot_type"] != "").sum()) if "shot_type" in df.columns else 0
        )

        self.logger.info(
            f"Stage 6 completed in {time.time() - stage_start:.2f}s - "
            f"Strokes: {num_strokes}, Shots: {num_shots}"
        )
        return df

    def _stage7_export(
        self,
        video_path: str,
        video_name: str,
        output_dir: Path,
        session: PipelineSession,
    ) -> Dict[str, str]:
        """Stage 7: Export CSV with keypoints and annotated video."""
        self.logger.info("Stage 7: Export and Visualization")
        stage_start = time.time()

        output_paths = {}
        df = session.current_df
        output_config = self.config.get("output", {})

        # Add keypoints to DataFrame (key difference from Pipeline)
        df = self._add_keypoints_to_dataframe(df, session.complex_data)

        # Export CSV (rally frames only)
        if output_config.get("save_csv", True):
            csv_filename = output_config.get("csv_filename", "{video_name}_annotations.csv")
            csv_path = output_dir / csv_filename.format(video_name=video_name)

            # Filter to rally frames only
            if "is_rally_frame" in df.columns:
                export_df = df[df["is_rally_frame"]].copy()
            else:
                export_df = df.copy()

            csv_index = output_config.get("csv_index", True)
            export_df.to_csv(csv_path, index=csv_index)
            output_paths["csv"] = str(csv_path)
            self.logger.info(f"CSV exported: {csv_path}")

        # Export annotated video
        if output_config.get("save_annotated_video", True):
            video_filename = output_config.get("video_filename", "{video_name}_annotated.mp4")
            video_output_path = output_dir / video_filename.format(video_name=video_name)
            self._render_annotated_video(video_path, video_output_path, session)
            output_paths["video"] = str(video_output_path)
            self.logger.info(f"Annotated video exported: {video_output_path}")

        self.logger.info(f"Stage 7 completed in {time.time() - stage_start:.2f}s")
        return output_paths

    def _add_keypoints_to_dataframe(
        self,
        df: pd.DataFrame,
        complex_data: Dict[str, List],
    ) -> pd.DataFrame:
        """Add keypoint columns to the DataFrame.

        This is the key difference from Pipeline - keypoints are saved to CSV.

        Args:
            df: DataFrame with tracking data
            complex_data: Dictionary containing player keypoints

        Returns:
            DataFrame with keypoint columns added
        """
        player_1_keypoints = complex_data.get("player_1_keypoints", [])
        player_2_keypoints = complex_data.get("player_2_keypoints", [])

        # Initialize keypoint columns with NaN
        for player_id in [1, 2]:
            for kp_name in KEYPOINT_NAMES:
                df[f"player_{player_id}_kp_{kp_name}_x"] = np.nan
                df[f"player_{player_id}_kp_{kp_name}_y"] = np.nan

        # Fill in keypoint data
        for frame_idx in df.index:
            # Player 1 keypoints
            if (
                frame_idx < len(player_1_keypoints)
                and player_1_keypoints[frame_idx] is not None
            ):
                kp_array = player_1_keypoints[frame_idx]
                if kp_array is not None and len(kp_array) >= len(KEYPOINT_NAMES):
                    for i, kp_name in enumerate(KEYPOINT_NAMES):
                        if i < len(kp_array):
                            x, y = kp_array[i][0], kp_array[i][1]
                            if x != 0 or y != 0:
                                df.loc[frame_idx, f"player_1_kp_{kp_name}_x"] = x
                                df.loc[frame_idx, f"player_1_kp_{kp_name}_y"] = y

            # Player 2 keypoints
            if (
                frame_idx < len(player_2_keypoints)
                and player_2_keypoints[frame_idx] is not None
            ):
                kp_array = player_2_keypoints[frame_idx]
                if kp_array is not None and len(kp_array) >= len(KEYPOINT_NAMES):
                    for i, kp_name in enumerate(KEYPOINT_NAMES):
                        if i < len(kp_array):
                            x, y = kp_array[i][0], kp_array[i][1]
                            if x != 0 or y != 0:
                                df.loc[frame_idx, f"player_2_kp_{kp_name}_x"] = x
                                df.loc[frame_idx, f"player_2_kp_{kp_name}_y"] = y

        return df

    def _render_annotated_video(
        self,
        video_path: str,
        output_path: Path,
        session: PipelineSession,
    ):
        """Render annotated video with clean visualizations."""
        df = session.current_df
        complex_data = session.complex_data

        if df is None or df.empty:
            self.logger.warning("No data to render")
            return

        viz_config = self.config.get("visualization", {})

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        codec = self.config.get("output", {}).get("video_codec", "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Filter to rally frames
        if "is_rally_frame" in df.columns:
            rally_frame_set = set(df[df["is_rally_frame"]].index.tolist())
        else:
            rally_frame_set = set(df.index.tolist())

        total_rally_frames = len(rally_frame_set)

        # Get complex data
        player_bboxes = {
            1: complex_data.get("player_1_bboxes", []),
            2: complex_data.get("player_2_bboxes", []),
        }
        player_keypoints = {
            1: complex_data.get("player_1_keypoints", []),
            2: complex_data.get("player_2_keypoints", []),
        }

        # Build hit frame lookup for persistent markers
        hit_marker_duration = viz_config.get("hit_marker_duration", 30)
        stroke_label_duration = viz_config.get("stroke_label_duration", 45)
        shot_label_duration = viz_config.get("shot_label_duration", 45)

        # Get hit frames
        wall_hit_frames = (
            df[df["is_wall_hit"] == 1].index.tolist()
            if "is_wall_hit" in df.columns
            else []
        )
        racket_hit_frames = (
            df[df["is_racket_hit"] == 1].index.tolist()
            if "is_racket_hit" in df.columns
            else []
        )

        frame_number = 0
        pbar = tqdm(
            total=total_rally_frames,
            desc="Rendering video",
            disable=not self.config.get("logging", {}).get("show_progress", True),
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number in rally_frame_set:
                frame_data = df.loc[frame_number]
                frame = self._draw_frame_annotations(
                    frame=frame,
                    frame_data=frame_data,
                    frame_idx=frame_number,
                    df=df,
                    player_bboxes=player_bboxes,
                    player_keypoints=player_keypoints,
                    viz_config=viz_config,
                    calibration=session.calibration,
                    wall_hit_frames=wall_hit_frames,
                    racket_hit_frames=racket_hit_frames,
                    hit_marker_duration=hit_marker_duration,
                    stroke_label_duration=stroke_label_duration,
                    shot_label_duration=shot_label_duration,
                )
                out.write(frame)
                pbar.update(1)

            frame_number += 1

        pbar.close()
        cap.release()
        out.release()

    def _draw_frame_annotations(
        self,
        frame: np.ndarray,
        frame_data: pd.Series,
        frame_idx: int,
        df: pd.DataFrame,
        player_bboxes: Dict,
        player_keypoints: Dict,
        viz_config: Dict,
        calibration: CourtCalibrationOutput,
        wall_hit_frames: List[int],
        racket_hit_frames: List[int],
        hit_marker_duration: int,
        stroke_label_duration: int,
        shot_label_duration: int,
    ) -> np.ndarray:
        """Draw all annotations on a single frame with clean layout."""
        # Get colors from config
        player_colors = {
            1: tuple(viz_config.get("player_colors", {}).get("player_1", [0, 255, 0])),
            2: tuple(viz_config.get("player_colors", {}).get("player_2", [255, 0, 0])),
        }
        ball_color = tuple(viz_config.get("ball_color", [0, 255, 255]))
        court_color = tuple(viz_config.get("court_line_color", [255, 255, 0]))
        wall_hit_color = tuple(viz_config.get("wall_hit_color", [0, 0, 255]))
        racket_hit_color = tuple(viz_config.get("racket_hit_color", [255, 0, 255]))
        rally_info_color = tuple(viz_config.get("rally_info_color", [255, 255, 255]))
        stroke_label_color = tuple(viz_config.get("stroke_label_color", [0, 255, 0]))
        shot_label_color = tuple(viz_config.get("shot_label_color", [255, 0, 255]))

        font = viz_config.get("font", cv2.FONT_HERSHEY_SIMPLEX)
        font_thickness = viz_config.get("font_thickness", 2)

        # 1. Draw court lines
        if viz_config.get("draw_court_lines", True) and calibration and calibration.court_keypoints:
            self._draw_court_overlay(frame, calibration.court_keypoints, court_color, viz_config)

        # 2. Draw ball trajectory
        if viz_config.get("draw_ball_trajectory", True):
            trajectory_length = viz_config.get("ball_trajectory_length", 30)
            trajectory_color = tuple(viz_config.get("ball_trajectory_color", [0, 200, 200]))
            self._draw_ball_trajectory(frame, df, frame_idx, trajectory_length, trajectory_color)

        # 3. Draw ball
        if viz_config.get("draw_ball", True):
            ball_x = frame_data.get("ball_x")
            ball_y = frame_data.get("ball_y")
            ball_radius = viz_config.get("ball_radius", 6)
            if pd.notna(ball_x) and pd.notna(ball_y):
                cv2.circle(frame, (int(ball_x), int(ball_y)), ball_radius, ball_color, -1)

        # 4. Draw players (bboxes, keypoints, skeletons)
        bbox_thickness = viz_config.get("bbox_thickness", 2)
        label_font_scale = viz_config.get("label_font_scale", 0.6)
        label_font_thickness = viz_config.get("label_font_thickness", 2)

        for player_id in [1, 2]:
            color = player_colors[player_id]
            bbox = None
            if frame_idx < len(player_bboxes[player_id]):
                bbox = player_bboxes[player_id][frame_idx]

            kp_data = None
            if frame_idx < len(player_keypoints[player_id]):
                kp_data = player_keypoints[player_id][frame_idx]

            pos_x = frame_data.get(f"player_{player_id}_x_pixel")

            if bbox is not None and pd.notna(pos_x):
                # Draw bounding box
                if viz_config.get("draw_player_boxes", True):
                    cv2.rectangle(
                        frame,
                        (int(bbox.x1), int(bbox.y1)),
                        (int(bbox.x2), int(bbox.y2)),
                        color,
                        bbox_thickness,
                    )
                    # Draw player ID label
                    cv2.putText(
                        frame,
                        f"P{player_id}",
                        (int(bbox.x1), int(bbox.y1) - 5),
                        font,
                        label_font_scale,
                        color,
                        label_font_thickness,
                    )

                # Draw keypoints and skeleton
                if kp_data is not None:
                    if viz_config.get("draw_player_skeleton", True):
                        self._draw_skeleton(frame, kp_data, color, viz_config)
                    if viz_config.get("draw_player_keypoints", True):
                        self._draw_keypoint_circles(frame, kp_data, color, viz_config)

        # 5. Draw hit markers (wall and racket)
        if viz_config.get("draw_hit_markers", True):
            hit_radius = viz_config.get("hit_marker_radius", 20)

            # Wall hits
            for hit_frame in wall_hit_frames:
                if hit_frame <= frame_idx < hit_frame + hit_marker_duration:
                    hit_data = df.loc[hit_frame]
                    hit_x, hit_y = hit_data.get("ball_x"), hit_data.get("ball_y")
                    if pd.notna(hit_x) and pd.notna(hit_y):
                        self._draw_hit_marker(frame, int(hit_x), int(hit_y), wall_hit_color, hit_radius, "W")

            # Racket hits
            for hit_frame in racket_hit_frames:
                if hit_frame <= frame_idx < hit_frame + hit_marker_duration:
                    hit_data = df.loc[hit_frame]
                    hit_x, hit_y = hit_data.get("ball_x"), hit_data.get("ball_y")
                    if pd.notna(hit_x) and pd.notna(hit_y):
                        self._draw_hit_marker(frame, int(hit_x), int(hit_y), racket_hit_color, hit_radius, "R")

        # 6. Draw info panel (top-left corner)
        y_offset = 30
        line_height = 25

        # Rally info
        if viz_config.get("show_rally_info", True):
            rally_id = frame_data.get("rally_id", -1)
            is_rally = frame_data.get("is_rally_frame", False)
            rally_font_scale = viz_config.get("rally_info_font_scale", 0.8)

            if pd.notna(rally_id) and rally_id >= 0:
                rally_text = f"Rally {int(rally_id)}"
            else:
                rally_text = "Between Rallies" if not is_rally else "Rally"

            self._draw_text_with_background(
                frame, rally_text, (15, y_offset), font, rally_font_scale, rally_info_color, font_thickness
            )
            y_offset += line_height

        # 7. Draw stroke label (if recent racket hit)
        if viz_config.get("show_stroke_labels", True):
            stroke_font_scale = viz_config.get("stroke_label_font_scale", 0.7)
            for hit_frame in racket_hit_frames:
                if hit_frame <= frame_idx < hit_frame + stroke_label_duration:
                    hit_data = df.loc[hit_frame]
                    stroke_type = hit_data.get("stroke_type", "")
                    player_id = hit_data.get("racket_hit_player_id")
                    if pd.notna(stroke_type) and stroke_type:
                        player_str = f"P{int(player_id)}" if pd.notna(player_id) else ""
                        stroke_text = f"Stroke: {stroke_type} {player_str}"
                        self._draw_text_with_background(
                            frame, stroke_text, (15, y_offset), font, stroke_font_scale, stroke_label_color, font_thickness
                        )
                        y_offset += line_height
                    break

        # 8. Draw shot label (if recent racket hit)
        if viz_config.get("show_shot_labels", True):
            shot_font_scale = viz_config.get("shot_label_font_scale", 0.7)
            for hit_frame in racket_hit_frames:
                if hit_frame <= frame_idx < hit_frame + shot_label_duration:
                    hit_data = df.loc[hit_frame]
                    shot_type = hit_data.get("shot_type", "")
                    if pd.notna(shot_type) and shot_type:
                        shot_text = f"Shot: {shot_type}"
                        self._draw_text_with_background(
                            frame, shot_text, (15, y_offset), font, shot_font_scale, shot_label_color, font_thickness
                        )
                        y_offset += line_height
                    break

        return frame

    def _draw_ball_trajectory(
        self,
        frame: np.ndarray,
        df: pd.DataFrame,
        frame_idx: int,
        trajectory_length: int,
        color: tuple,
    ):
        """Draw ball trajectory as fading line."""
        start_idx = max(0, frame_idx - trajectory_length)
        for i in range(start_idx, frame_idx):
            if i in df.index and i + 1 in df.index:
                p1_x, p1_y = df.loc[i, "ball_x"], df.loc[i, "ball_y"]
                p2_x, p2_y = df.loc[i + 1, "ball_x"], df.loc[i + 1, "ball_y"]
                if pd.notna(p1_x) and pd.notna(p1_y) and pd.notna(p2_x) and pd.notna(p2_y):
                    # Fade effect (older = more transparent)
                    alpha = (i - start_idx + 1) / trajectory_length
                    faded_color = tuple(int(c * alpha) for c in color)
                    thickness = max(1, int(2 * alpha))
                    cv2.line(frame, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), faded_color, thickness)

    def _draw_hit_marker(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        color: tuple,
        radius: int,
        label: str,
    ):
        """Draw a clean hit marker (circle with label)."""
        # Outer circle
        cv2.circle(frame, (x, y), radius, color, 2)
        # Inner circle
        cv2.circle(frame, (x, y), radius // 2, color, -1)
        # Label
        cv2.putText(
            frame, label, (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        font: int,
        font_scale: float,
        color: tuple,
        thickness: int,
    ):
        """Draw text with semi-transparent background for readability."""
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x, y = position
        padding = 5

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - padding, y - text_size[1] - padding),
            (x + text_size[0] + padding, y + padding),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints_array: np.ndarray,
        color: tuple,
        viz_config: Dict,
    ):
        """Draw skeleton connections."""
        if keypoints_array is None or len(keypoints_array) == 0:
            return

        skeleton_thickness = viz_config.get("skeleton_thickness", 2)

        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start_body_idx = start_idx - 5
            end_body_idx = end_idx - 5

            if 0 <= start_body_idx < len(keypoints_array) and 0 <= end_body_idx < len(keypoints_array):
                x1, y1 = keypoints_array[start_body_idx][0], keypoints_array[start_body_idx][1]
                x2, y2 = keypoints_array[end_body_idx][0], keypoints_array[end_body_idx][1]

                if (x1 != 0 or y1 != 0) and (x2 != 0 or y2 != 0):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, skeleton_thickness)

    def _draw_keypoint_circles(
        self,
        frame: np.ndarray,
        keypoints_array: np.ndarray,
        color: tuple,
        viz_config: Dict,
    ):
        """Draw keypoint circles."""
        if keypoints_array is None or len(keypoints_array) == 0:
            return

        keypoint_radius = viz_config.get("keypoint_radius", 4)

        for i in range(len(keypoints_array)):
            x, y = keypoints_array[i][0], keypoints_array[i][1]
            if x != 0 or y != 0:
                cv2.circle(frame, (int(x), int(y)), keypoint_radius, color, -1)

    def _draw_court_overlay(
        self,
        frame: np.ndarray,
        court_keypoints: Dict[str, Any],
        color: tuple,
        viz_config: Dict,
    ):
        """Draw court calibration overlay."""
        court_thickness = viz_config.get("court_thickness", 2)

        # Group keypoints by class
        keypoints_by_class = {}
        for key, point in court_keypoints.items():
            parts = key.rsplit("_", 2)
            if len(parts) >= 3:
                class_name = parts[0]
                corner_type = f"{parts[-2]}_{parts[-1]}"
            else:
                continue

            if class_name not in keypoints_by_class:
                keypoints_by_class[class_name] = {}
            keypoints_by_class[class_name][corner_type] = point

        # Draw each class
        for class_name, corners in keypoints_by_class.items():
            points = []
            for corner_type in ["top_left", "top_right", "bottom_right", "bottom_left"]:
                if corner_type in corners:
                    point = corners[corner_type]
                    points.append((int(point.x), int(point.y)))

            if len(points) == 4:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, color, court_thickness)

            for point in points:
                cv2.circle(frame, point, 3, color, -1)


if __name__ == "__main__":
    """Run the annotator with configuration from annotation.yaml."""
    annotator = Annotator()
    output_paths = annotator.run()

    print("\nAnnotation completed successfully!")
    print("Generated outputs:")
    for output_type, path in output_paths.items():
        print(f"  {output_type.upper()}: {path}")
