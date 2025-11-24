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
from typing import Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import json

import cv2
import numpy as np
import pandas as pd
import torch
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
    RallySegment,
    WallHitDetectionInput,
    RacketHitDetectionInput,
    StrokeClassificationInput,
    ShotClassificationInput,
    PointWinDetectionInput,
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
from squashcopilot.modules.point_win_detection.point_win_detector import (
    PointWinDetector,
)
from squashcopilot.pipeline.frame_reader import BatchFrameReader


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

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path: Path to custom pipeline config YAML. If None, uses default config.
            config: Optional config dictionary. If provided, overrides values from config file.
            progress_callback: Optional callback function(stage: str, percent: float) for progress updates.
        """
        # Load base config
        if config_path:
            self.config = load_config(config_path=config_path)
        else:
            self.config = load_config(config_name="pipeline")

        # Override with provided config dict
        if config:
            self._merge_config(config)

        # Store progress callback
        self.progress_callback = progress_callback

        self._setup_logging()
        self.logger.info("Initializing pipeline modules...")
        self._initialize_modules()
        self.logger.info("Pipeline initialized successfully")

    def _merge_config(self, override: Dict) -> None:
        """Deep merge override config into base config."""
        def merge(base: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge(base[key], value)
                else:
                    base[key] = value
            return base
        merge(self.config, override)

    def _report_progress(self, stage: str, percent: float) -> None:
        """Report progress to callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(stage, percent)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")

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

        # Point Win Detection
        if module_configs.get("point_win_detection", {}).get("enabled", True):
            self.point_win_detector = PointWinDetector()
        else:
            self.point_win_detector = None

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
        self._report_progress("court_calibration", 5)
        session.calibration = self._stage1_calibrate_court(video_path)
        self._report_progress("court_calibration", 10)

        # Stage 2: Frame-by-frame Tracking -> DataFrame
        self._report_progress("tracking", 10)
        session.current_df, session.complex_data = self._stage2_track_frames(
            video_path, session.calibration, video_metadata
        )
        self._report_progress("tracking", 50)

        # Stage 3: Trajectory Postprocessing
        self._report_progress("postprocessing", 50)
        session.current_df, session.complex_data = self._stage3_postprocess(
            session.current_df, session.complex_data
        )
        self._report_progress("postprocessing", 60)

        # Stage 4: Rally Segmentation
        self._report_progress("rally_segmentation", 60)
        session.current_df, session.rally_segments = self._stage4_segment_rallies(
            session.current_df, session.calibration
        )
        self._report_progress("rally_segmentation", 70)

        # Stage 5: Hit Detection
        self._report_progress("hit_detection", 70)
        session.current_df = self._stage5_detect_hits(
            session.current_df, session.rally_segments, session.calibration
        )
        self._report_progress("hit_detection", 75)

        # Stage 6: Stroke and Shot Classification
        self._report_progress("classification", 75)
        session.current_df = self._stage6_classify(
            session.current_df,
            session.complex_data,
            session.rally_segments,
            session.calibration,
        )
        self._report_progress("classification", 80)

        # Stage 7: Point Win Detection
        self._report_progress("point_win_detection", 80)
        session.current_df = self._stage7_detect_point_winners(
            session.current_df, session.rally_segments, session.calibration
        )
        self._report_progress("point_win_detection", 85)

        # Stage 8: Export Results
        self._report_progress("export", 85)
        output_paths = self._stage8_export(
            video_path=video_path,
            video_name=video_name,
            output_dir=output_dir,
            session=session,
        )
        self._report_progress("completed", 100)

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

    def _process_ball_batch(
        self,
        images: List[np.ndarray],
        frame_numbers: List[int],
        timestamps: List[float],
        batch_size: int,
        carryover_frames: Optional[List[np.ndarray]],
    ) -> Tuple[List[BallTrackingOutput], Optional[List[np.ndarray]]]:
        """Process ball tracking for a batch (helper for parallel execution)."""
        if self.ball_tracker:
            return self.ball_tracker.process_batch(
                frames=images,
                frame_numbers=frame_numbers,
                timestamps=timestamps,
                batch_size=batch_size,
                carryover_frames=carryover_frames,
            )
        else:
            # Create empty outputs if ball tracker disabled
            empty_outputs = [
                BallTrackingOutput(
                    frame_number=fn,
                    timestamp=ts,
                    ball_detected=False,
                    ball_x=None,
                    ball_y=None,
                    ball_confidence=0.0,
                )
                for fn, ts in zip(frame_numbers, timestamps)
            ]
            return empty_outputs, carryover_frames

    def _process_player_batch(
        self,
        images: List[np.ndarray],
        frame_numbers: List[int],
        timestamps: List[float],
        batch_size: int,
    ) -> List[PlayerTrackingOutput]:
        """Process player tracking for a batch (helper for parallel execution)."""
        if self.player_tracker:
            return self.player_tracker.process_batch(
                frames=images,
                frame_numbers=frame_numbers,
                timestamps=timestamps,
                batch_size=batch_size,
            )
        else:
            # Create empty outputs if player tracker disabled
            return [
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
                for fn, ts in zip(frame_numbers, timestamps)
            ]

    def _stage2_track_frames(
        self,
        video_path: str,
        calibration: CourtCalibrationOutput,
        video_metadata: VideoMetadata,
    ) -> tuple:
        """Stage 2: Frame-by-frame tracking -> DataFrame.

        Uses batch processing for both ball and player tracking to optimize GPU utilization.
        Ball and player tracking run in parallel using CUDA streams for true GPU parallelism.

        CUDA Streams Strategy:
        - Each tracker gets its own CUDA stream
        - Streams allow overlapping GPU memory transfers and kernel execution
        - ThreadPoolExecutor handles CPU preprocessing overlap
        - Combined effect: both trackers can utilize GPU simultaneously
        """
        self.logger.info(
            "Stage 2: Frame-by-Frame Tracking (using CUDA streams for parallel GPU execution)"
        )
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

        # Check if CUDA is available for stream-based parallelism
        use_cuda_streams = torch.cuda.is_available()

        if use_cuda_streams:
            # Create separate CUDA streams for ball and player tracking
            # This allows overlapping GPU operations between the two trackers
            ball_stream = torch.cuda.Stream()
            player_stream = torch.cuda.Stream()

            # Assign streams to trackers so their inference uses the correct stream
            if self.ball_tracker and hasattr(self.ball_tracker, "tracker"):
                self.ball_tracker.tracker._cuda_stream = ball_stream
            if self.player_tracker:
                self.player_tracker._cuda_stream = player_stream
        else:
            ball_stream = None
            player_stream = None
            self.logger.info("CUDA not available, using sequential execution")

        # Use ThreadPoolExecutor for CPU preprocessing overlap
        # Combined with CUDA streams, this allows:
        # 1. CPU preprocessing for ball tracking overlaps with GPU inference for player tracking
        # 2. CPU preprocessing for player tracking overlaps with GPU inference for ball tracking
        # 3. GPU operations from both trackers can overlap via different streams
        with ThreadPoolExecutor(max_workers=2) as executor:
            for batch in frame_reader:
                # Submit both tasks in parallel to ThreadPoolExecutor
                # Each tracker will use its assigned CUDA stream internally
                ball_future = executor.submit(
                    self._process_ball_batch,
                    batch.images,
                    batch.frame_numbers,
                    batch.timestamps,
                    batch_size,
                    ball_carryover_frames,
                )
                player_future = executor.submit(
                    self._process_player_batch,
                    batch.images,
                    batch.frame_numbers,
                    batch.timestamps,
                    batch_size,
                )

                # Wait for both to complete
                batch_ball_outputs, ball_carryover_frames = ball_future.result()
                batch_player_outputs = player_future.result()

                if use_cuda_streams:
                    # Synchronize CUDA to ensure all GPU operations are complete
                    # before processing next batch
                    torch.cuda.synchronize()

                ball_outputs.extend(batch_ball_outputs)
                player_outputs.extend(batch_player_outputs)

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

    def _stage7_detect_point_winners(
        self,
        df: pd.DataFrame,
        segments: List[RallySegment],
        calibration: CourtCalibrationOutput,
    ) -> pd.DataFrame:
        """Stage 7: Point Win Detection.

        Detects point winners for each rally using serve sequence analysis and ball physics.
        """
        self.logger.info("Stage 7: Point Win Detection")
        stage_start = time.time()

        if self.point_win_detector:
            point_win_input = PointWinDetectionInput(
                df=df, segments=segments, calibration=calibration
            )
            point_win_output = self.point_win_detector.detect_point_winners(point_win_input)
            df = point_win_output.df

            self.logger.info(
                f"Stage 7 completed in {time.time() - stage_start:.2f}s - "
                f"Rallies: {point_win_output.num_rallies}, "
                f"Lets: {point_win_output.num_lets}, "
                f"Unknown: {point_win_output.num_unknown}"
            )
        else:
            self.logger.info("Point win detection disabled, skipping")

        return df

    def _stage8_export(
        self,
        video_path: str,
        video_name: str,
        output_dir: Path,
        session: PipelineSession,
    ) -> Dict[str, str]:
        """Stage 8: Export CSV, annotated video, and statistics."""
        self.logger.info("Stage 8: Export and Visualization")
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

        self.logger.info(f"Stage 8 completed in {time.time() - stage_start:.2f}s")
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
