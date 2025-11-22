"""
Complete pipeline orchestrator for squash video analysis.

This module provides the Pipeline class that coordinates all stages of video processing
from raw input video to comprehensive analysis outputs (CSV, annotated video, statistics).

Uses DataFrame-based data flow architecture where each stage progressively enriches
the DataFrame with new columns.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from squashcopilot.common.types import Frame
from squashcopilot.common.utils import load_config
from squashcopilot.common.models import (
    VideoMetadata,
    CourtCalibrationInput,
    CourtCalibrationOutput,
    PlayerTrackingOutput,
    player_tracking_outputs_to_dataframe,
    PlayerPostprocessingInput,
    PlayerPostprocessingOutput,
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
from squashcopilot.pipeline.frame_reader import BatchFrameReader, FrameBatch


class Pipeline:
    """
    Complete pipeline orchestrator for squash video analysis.

    Coordinates all 7 stages of processing with DataFrame-based data flow:
    1. Court calibration (first frame)
    2. Frame-by-frame tracking (player + ball) -> DataFrame
    3. Trajectory postprocessing (smoothing, interpolation)
    4. Rally segmentation (identify rally boundaries)
    5. Hit detection (wall hits + racket hits)
    6. Stroke and shot classification
    7. Export (CSV, annotated video, statistics JSON)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path: Path to custom pipeline config YAML. If None, uses default config.
        """
        if config_path:
            self.config = load_config(config_path=config_path)
        else:
            self.config = load_config(config_name="pipeline")

        self._setup_logging()
        self.logger.info("Initializing pipeline modules...")
        self._initialize_modules()
        self.logger.info("Pipeline initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        # Get or create logger for this module
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.logger.propagate = (
            False  # Don't propagate to root logger (avoids duplicate messages)
        )

        # Only add handler if logger doesn't have one yet
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler = (
                logging.FileHandler(log_config["log_file"])
                if log_config.get("log_file")
                else logging.StreamHandler()
            )
            handler.setLevel(level)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _initialize_modules(self):
        """Initialize all processing modules."""
        module_configs = self.config.get("modules", {})

        # Court Calibration
        if module_configs.get("court_calibration", {}).get("enabled", True):
            self.court_calibrator = CourtCalibrator()
        else:
            self.court_calibrator = None

        # Player Tracking
        if module_configs.get("player_tracking", {}).get("enabled", True):
            self.player_tracker = PlayerTracker()
        else:
            self.player_tracker = None

        # Ball Tracking
        if module_configs.get("ball_tracking", {}).get("enabled", True):
            self.ball_tracker = BallTracker()
            self.wall_hit_detector = WallHitDetector()
            self.racket_hit_detector = RacketHitDetector()
        else:
            self.ball_tracker = None
            self.wall_hit_detector = None
            self.racket_hit_detector = None

        # Rally State Detection
        if module_configs.get("rally_state_detection", {}).get("enabled", True):
            self.rally_detector = RallyStateDetector()
        else:
            self.rally_detector = None

        # Stroke Detection
        if module_configs.get("stroke_detection", {}).get("enabled", True):
            self.stroke_detector = StrokeDetector()
        else:
            self.stroke_detector = None

        # Shot Classification
        if module_configs.get("shot_type_classification", {}).get("enabled", True):
            self.shot_classifier = ShotClassifier()
        else:
            self.shot_classifier = None

    def run(self) -> Dict[str, str]:
        """
        Run the complete pipeline on a video.

        Returns:
            Dictionary with paths to generated outputs
        """
        video_path = self.config.get("video_path")
        if not video_path:
            raise ValueError("video_path not specified in pipeline config")

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Starting pipeline for video: {video_path}")
        start_time = time.time()

        # Setup output directory
        video_name = Path(video_path).stem
        base_dir = self.config["output"]["base_directory"]
        if self.config["output"]["create_video_subdirectory"]:
            output_dir = Path(base_dir) / video_name
        else:
            output_dir = Path(base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video metadata
        video_metadata = self._get_video_metadata(video_path)
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
        session.calibration = self._stage1_calibrate_court(video_path)

        # Stage 2: Frame-by-frame Tracking -> DataFrame
        session.current_df, session.complex_data = self._stage2_track_frames(
            video_path, session.calibration, video_metadata
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

        # Stage 7: Export Results
        output_paths = self._stage7_export(
            video_path=video_path,
            video_name=video_name,
            output_dir=output_dir,
            session=session,
        )

        total_time = time.time() - start_time
        self.logger.info(f"Pipeline completed in {total_time:.2f}s")

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

        # Apply max_seconds limit if specified
        max_seconds = self.config.get("max_seconds")
        if max_seconds is not None:
            max_frames = int(max_seconds * fps)
            if max_frames < total_frames:
                self.logger.info(f"Limiting to {max_seconds}s ({max_frames} frames)")
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

        if self.court_calibrator is None:
            self.logger.warning("Court calibration disabled")
            return None

        cap = cv2.VideoCapture(video_path)
        ret, frame_img = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read first frame from {video_path}")

        frame = Frame(image=frame_img, frame_number=0, timestamp=0.0)
        calib_input = CourtCalibrationInput(frame=frame)
        calibration = self.court_calibrator.process_frame(calib_input)

        self.logger.info(f"Stage 1 completed in {time.time() - stage_start:.2f}s")
        return calibration

    def _stage2_track_frames(
        self,
        video_path: str,
        calibration: CourtCalibrationOutput,
        video_metadata: VideoMetadata,
    ) -> tuple:
        """Stage 2: Frame-by-frame tracking -> DataFrame.

        Uses batch processing for ball tracking to optimize GPU utilization.
        Player tracking still processes frame-by-frame (to be optimized in Phase 2).
        """
        self.logger.info("Stage 2: Frame-by-Frame Tracking (using batch processing)")
        stage_start = time.time()

        total_frames = video_metadata.total_frames
        fps = video_metadata.fps

        # Get batch processing config
        batch_size = self.config["processing"].get("batch_size", 32)
        prefetch_batches = self.config["processing"].get("prefetch_batches", 2)

        player_outputs = []
        ball_outputs = []

        # Set ball tracker for black ball if needed
        if self.ball_tracker and calibration:
            self.ball_tracker.set_is_black_ball(calibration.is_black_ball)

        # Set player tracker calibration
        if self.player_tracker and calibration:
            self.player_tracker.set_calibration(calibration)

        # Create batch frame reader
        frame_reader = BatchFrameReader(
            video_path=video_path,
            batch_size=batch_size,
            max_frames=total_frames,
            fps=fps,
            prefetch=True,
            prefetch_batches=prefetch_batches,
        )

        # Progress bar for total frames
        pbar = tqdm(
            total=total_frames,
            desc="Tracking frames",
            disable=not self.config["logging"]["show_progress"],
        )

        # Carryover frames for ball tracking continuity across batches
        ball_carryover_frames = None

        for batch in frame_reader:
            # Process ball tracking in batch (GPU optimized)
            if self.ball_tracker:
                batch_ball_outputs, ball_carryover_frames = (
                    self.ball_tracker.process_batch(
                        frames=batch.images,
                        frame_numbers=batch.frame_numbers,
                        timestamps=batch.timestamps,
                        batch_size=batch_size,
                        carryover_frames=ball_carryover_frames,
                    )
                )
                ball_outputs.extend(batch_ball_outputs)
            else:
                # Create empty outputs if ball tracker disabled
                for fn, ts in zip(batch.frame_numbers, batch.timestamps):
                    ball_outputs.append(
                        BallTrackingOutput(
                            frame_number=fn,
                            timestamp=ts,
                            ball_detected=False,
                            ball_x=None,
                            ball_y=None,
                            ball_confidence=0.0,
                        )
                    )

            # Process player tracking in batch (GPU optimized)
            if self.player_tracker:
                batch_player_outputs = self.player_tracker.process_batch(
                    frames=batch.images,
                    frame_numbers=batch.frame_numbers,
                    timestamps=batch.timestamps,
                    batch_size=batch_size,
                )
                player_outputs.extend(batch_player_outputs)
            else:
                # Create empty outputs if player tracker disabled
                for fn, ts in zip(batch.frame_numbers, batch.timestamps):
                    player_outputs.append(
                        PlayerTrackingOutput(
                            frame_number=fn,
                            timestamp=ts,
                            player_1_detected=False,
                            player_1_x_pixel=None,
                            player_1_y_pixel=None,
                            player_1_x_meter=None,
                            player_1_y_meter=None,
                            player_1_confidence=0.0,
                            player_1_bbox=None,
                            player_1_keypoints=None,
                            player_2_detected=False,
                            player_2_x_pixel=None,
                            player_2_y_pixel=None,
                            player_2_x_meter=None,
                            player_2_y_meter=None,
                            player_2_confidence=0.0,
                            player_2_bbox=None,
                            player_2_keypoints=None,
                        )
                    )

            pbar.update(len(batch))

        pbar.close()

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
        if self.player_tracker:
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
            complex_data["player_1_bboxes"] = (
                player_postprocess_output.player_bboxes.get(1, [])
            )
            complex_data["player_2_bboxes"] = (
                player_postprocess_output.player_bboxes.get(2, [])
            )

        # Postprocess ball trajectory
        if self.ball_tracker:
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

        if self.rally_detector is None:
            self.logger.warning("Rally detection disabled")
            df["is_rally_frame"] = True
            df["rally_id"] = 0
            return df, []

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
        if self.wall_hit_detector and segments:
            wall_hit_input = WallHitDetectionInput(
                df=df, segments=segments, calibration=calibration
            )
            wall_hit_output = self.wall_hit_detector.detect_wall_hits(wall_hit_input)
            df = wall_hit_output.df

        # Racket hit detection
        if self.racket_hit_detector and segments and "is_wall_hit" in df.columns:
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
        if self.stroke_detector:
            stroke_input = StrokeClassificationInput(
                df=df, player_keypoints=player_keypoints
            )
            batch_size = self.config["processing"].get("batch_size", 32)
            stroke_output = self.stroke_detector.detect_strokes(
                stroke_input, batch_size=batch_size
            )
            df = stroke_output.df

        # Shot classification
        if self.shot_classifier:
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
        """Stage 7: Export CSV, annotated video, and statistics."""
        self.logger.info("Stage 7: Export and Visualization")
        stage_start = time.time()

        output_paths = {}
        df = session.current_df

        # Export CSV (rally frames only)
        if self.config["output"]["save_csv"]:
            csv_filename = self.config["output"]["csv_filename"].format(
                video_name=video_name
            )
            csv_path = output_dir / csv_filename

            # Filter to rally frames only
            if "is_rally_frame" in df.columns:
                export_df = df[df["is_rally_frame"]].copy()
            else:
                export_df = df.copy()

            # Filter columns if specified in config
            csv_columns = self.config["output"].get("csv_columns")
            if csv_columns:
                # Only include columns that exist in the DataFrame
                available_columns = [
                    col for col in csv_columns if col in export_df.columns
                ]
                missing_columns = [
                    col for col in csv_columns if col not in export_df.columns
                ]
                if missing_columns:
                    self.logger.warning(
                        f"CSV columns not found in data: {missing_columns}"
                    )
                export_df = export_df[available_columns]

            export_df.to_csv(csv_path)
            output_paths["csv"] = str(csv_path)
            self.logger.info(
                f"CSV exported: {csv_path} ({len(export_df.columns)} columns)"
            )

        # Export statistics JSON
        if self.config["output"]["save_statistics"]:
            stats = self._compute_statistics(video_name, session)
            stats_filename = self.config["output"]["stats_filename"].format(
                video_name=video_name
            )
            stats_path = output_dir / stats_filename
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            output_paths["stats"] = str(stats_path)
            self.logger.info(f"Statistics exported: {stats_path}")

        # Export annotated video
        if self.config["output"]["save_annotated_video"]:
            video_filename = self.config["output"]["video_filename"].format(
                video_name=video_name
            )
            video_output_path = output_dir / video_filename
            self._render_annotated_video(video_path, video_output_path, session)
            output_paths["video"] = str(video_output_path)
            self.logger.info(f"Annotated video exported: {video_output_path}")

        self.logger.info(f"Stage 7 completed in {time.time() - stage_start:.2f}s")
        return output_paths

    def _compute_statistics(self, video_name: str, session: PipelineSession) -> Dict:
        """Compute high-level statistics."""
        df = session.current_df
        video_metadata = session.video_metadata
        rally_segments = session.rally_segments

        stats = {
            "video_info": {
                "filename": video_name,
                "duration_seconds": video_metadata.duration_seconds,
                "fps": video_metadata.fps,
                "total_frames": video_metadata.total_frames,
                "resolution": [video_metadata.width, video_metadata.height],
            },
            "rallies": {
                "total_rallies": len(rally_segments),
                "total_play_time_seconds": sum(
                    s.num_frames / video_metadata.fps for s in rally_segments
                ),
            },
            "hits": {
                "wall_hits": (
                    int(df["is_wall_hit"].sum()) if "is_wall_hit" in df.columns else 0
                ),
                "racket_hits": (
                    int(df["is_racket_hit"].sum())
                    if "is_racket_hit" in df.columns
                    else 0
                ),
            },
            "strokes": {},
            "shots": {},
        }

        # Stroke statistics
        if "stroke_type" in df.columns:
            stroke_counts = (
                df[df["stroke_type"] != ""]["stroke_type"].value_counts().to_dict()
            )
            stats["strokes"] = stroke_counts

        # Shot statistics
        if "shot_type" in df.columns:
            shot_counts = (
                df[df["shot_type"] != ""]["shot_type"].value_counts().to_dict()
            )
            stats["shots"] = shot_counts

        return stats

    def _render_annotated_video(
        self,
        video_path: str,
        output_path: Path,
        session: PipelineSession,
    ):
        """Render annotated video with visualizations."""
        df = session.current_df

        if df is None or df.empty:
            self.logger.warning("No data to render")
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(
            *self.config["processing"]["video_writer_codec"]
        )
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Filter to rally frames
        if "is_rally_frame" in df.columns:
            rally_frame_set = set(df[df["is_rally_frame"]].index.tolist())
        else:
            rally_frame_set = set(df.index.tolist())

        total_rally_frames = len(rally_frame_set)

        frame_number = 0
        pbar = tqdm(
            total=total_rally_frames,
            desc="Rendering video",
            disable=not self.config["logging"]["show_progress"],
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number in rally_frame_set:
                frame_data = df.loc[frame_number]
                frame = self._draw_frame_annotations(frame, frame_data)
                out.write(frame)
                pbar.update(1)

            frame_number += 1

        pbar.close()
        cap.release()
        out.release()

    def _draw_frame_annotations(
        self, frame: np.ndarray, frame_data: pd.Series
    ) -> np.ndarray:
        """Draw annotations on a single frame."""
        vis_config = self.config["visualization"]

        # Draw players
        for player_id in [1, 2]:
            x = frame_data.get(f"player_{player_id}_x_pixel")
            y = frame_data.get(f"player_{player_id}_y_pixel")

            if pd.notna(x) and pd.notna(y):
                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), 10, tuple(vis_config["player_box_color"]), -1)
                cv2.putText(
                    frame,
                    f"P{player_id}",
                    (x - 20, y - 20),
                    vis_config["font"],
                    0.6,
                    tuple(vis_config["player_box_color"]),
                    vis_config["font_thickness"],
                )

        # Draw ball
        if vis_config["draw_ball"]:
            ball_x = frame_data.get("ball_x")
            ball_y = frame_data.get("ball_y")
            if pd.notna(ball_x) and pd.notna(ball_y):
                cv2.circle(
                    frame,
                    (int(ball_x), int(ball_y)),
                    vis_config["ball_radius"],
                    tuple(vis_config["ball_color"]),
                    -1,
                )

        # Draw rally ID
        if vis_config["show_rally_id"]:
            rally_id = frame_data.get("rally_id")
            if pd.notna(rally_id) and rally_id >= 0:
                cv2.putText(
                    frame,
                    f"Rally {int(rally_id)}",
                    tuple(vis_config["rally_id_position"]),
                    vis_config["font"],
                    vis_config["rally_id_font_scale"],
                    tuple(vis_config["rally_id_color"]),
                    vis_config["font_thickness"],
                )

        # Draw stroke label
        if vis_config["show_stroke_labels"]:
            stroke_type = frame_data.get("stroke_type")
            if pd.notna(stroke_type) and stroke_type:
                cv2.putText(
                    frame,
                    f"Stroke: {stroke_type}",
                    (20, 100),
                    vis_config["font"],
                    vis_config["stroke_label_font_scale"],
                    tuple(vis_config["stroke_label_color"]),
                    vis_config["font_thickness"],
                )

        # Draw shot label
        if vis_config["show_shot_labels"]:
            shot_type = frame_data.get("shot_type")
            if pd.notna(shot_type) and shot_type:
                cv2.putText(
                    frame,
                    f"Shot: {shot_type}",
                    (20, 150),
                    vis_config["font"],
                    vis_config["shot_label_font_scale"],
                    tuple(vis_config["shot_label_color"]),
                    vis_config["font_thickness"],
                )

        return frame


if __name__ == "__main__":
    """Run the pipeline with configuration from pipeline.yaml."""
    pipeline = Pipeline()
    output_paths = pipeline.run()

    print("\nPipeline completed successfully!")
    print("Generated outputs:")
    for output_type, path in output_paths.items():
        print(f"  {output_type.upper()}: {path}")
