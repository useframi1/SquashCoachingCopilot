"""
Complete pipeline orchestrator for squash video analysis.

This module provides the Pipeline class that coordinates all stages of video processing
from raw input video to comprehensive analysis outputs (CSV, annotated video, statistics).
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from squashcopilot.common.types import Frame
from squashcopilot.common.utils import load_config
from squashcopilot.common.models.court import (
    CourtCalibrationInput,
    CourtCalibrationResult,
)
from squashcopilot.common.models.player import (
    PlayerTrackingInput,
    PlayerTrackingResult,
    PlayerDetectionResult,
    PlayerPostprocessingInput,
    PlayerTrajectory,
)
from squashcopilot.common.models.ball import (
    BallTrackingInput,
    BallDetectionResult,
    BallPostprocessingInput,
    BallTrajectory,
    WallHitInput,
    WallHitDetectionResult,
    RacketHitInput,
    RacketHitDetectionResult,
    WallHit,
    RacketHit,
)
from squashcopilot.common.models.rally import RallySegmentationInput, RallySegment
from squashcopilot.common.models.stroke import (
    StrokeDetectionInput,
    StrokeResult,
    StrokeDetectionResult,
)
from squashcopilot.common.models.shot import (
    ShotClassificationInput,
    ShotResult,
    ShotClassificationResult,
)

from squashcopilot.modules.court_calibration.court_calibrator import CourtCalibrator
from squashcopilot.modules.player_tracking.player_tracker import PlayerTracker
from squashcopilot.modules.ball_tracking.ball_tracker import BallTracker
from squashcopilot.modules.ball_tracking.wall_hit_detector import WallHitDetector
from squashcopilot.modules.ball_tracking.racket_hit_detector import RacketHitDetector
from squashcopilot.modules.rally_state_detection.rally_state_detector import (
    RallyStateDetector,
)
from squashcopilot.modules.stroke_detection.stroke_detector import StrokeDetector
from squashcopilot.modules.shot_type_classification.shot_classifier import (
    ShotClassifier,
)


class Pipeline:
    """
    Complete pipeline orchestrator for squash video analysis.

    Coordinates all 7 stages of processing:
    1. Court calibration (first frame)
    2. Frame-by-frame tracking (player + ball)
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
        # Load configuration
        if config_path:
            self.config = load_config(config_path=config_path)
        else:
            self.config = load_config(config_name="pipeline")

        # Setup logging
        self._setup_logging()

        # Initialize all modules
        self.logger.info("Initializing pipeline modules...")
        self._initialize_modules()

        self.logger.info("Pipeline initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                (
                    logging.FileHandler(log_config["log_file"])
                    if log_config.get("log_file")
                    else logging.StreamHandler()
                )
            ],
        )
        self.logger = logging.getLogger(__name__)

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

        Uses video_path from config. Output directory is determined by config settings.

        Returns:
            Dictionary with paths to generated outputs:
            {
                "csv": path to CSV file,
                "video": path to annotated video,
                "stats": path to statistics JSON
            }
        """
        # Get video path from config
        video_path = self.config.get("video_path")
        if not video_path:
            raise ValueError("video_path not specified in pipeline config")

        # Verify file exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Starting pipeline for video: {video_path}")
        start_time = time.time()

        # Setup output directory from config
        video_name = Path(video_path).stem
        base_dir = self.config["output"]["base_directory"]
        if self.config["output"]["create_video_subdirectory"]:
            output_dir = Path(base_dir) / video_name
        else:
            output_dir = Path(base_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {output_dir}")

        # Get video metadata
        video_info = self._get_video_info(video_path)
        self.logger.info(
            f"Video info: {video_info['total_frames']} frames, "
            f"{video_info['fps']:.2f} FPS, {video_info['duration']:.2f}s"
        )

        # Stage 1: Court Calibration
        calibration_result = self._stage1_calibrate_court(video_path)

        # Stage 2: Frame-by-frame Tracking
        raw_player_data, raw_ball_data = self._stage2_track_frames(
            video_path, calibration_result, video_info
        )

        # Stage 3: Trajectory Postprocessing
        player_trajectories, ball_trajectory = self._stage3_postprocess(
            raw_player_data, raw_ball_data
        )

        # Stage 4: Rally Segmentation
        rally_segments = self._stage4_segment_rallies(
            ball_trajectory, player_trajectories, video_info
        )

        # Stage 5: Hit Detection
        wall_hits, racket_hits = self._stage5_detect_hits(
            ball_trajectory, player_trajectories, rally_segments
        )

        # Stage 6: Stroke and Shot Classification
        stroke_results, shot_results = self._stage6_classify(
            player_trajectories,
            racket_hits.racket_hits,
            wall_hits.wall_hits,
            video_info,
        )

        # Stage 7: Export Results
        output_paths = self._stage7_export(
            video_path=video_path,
            video_name=video_name,
            video_info=video_info,
            output_dir=output_dir,
            calibration_result=calibration_result,
            player_trajectories=player_trajectories,
            ball_trajectory=ball_trajectory,
            rally_segments=rally_segments,
            wall_hits=wall_hits.wall_hits,
            racket_hits=racket_hits.racket_hits,
            stroke_results=stroke_results.strokes,
            shot_results=shot_results.shots,
        )

        total_time = time.time() - start_time
        self.logger.info(f"Pipeline completed in {total_time:.2f}s")
        self.logger.info(f"Outputs: {output_paths}")

        return output_paths

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        # Validate FPS
        if info["fps"] == 0:
            cap.release()
            raise ValueError(
                f"Video FPS is 0. The video file may be corrupted or in an unsupported format: {video_path}"
            )

        # Validate other properties
        if info["total_frames"] == 0:
            cap.release()
            raise ValueError(f"Video has 0 frames: {video_path}")

        if info["width"] == 0 or info["height"] == 0:
            cap.release()
            raise ValueError(f"Video has invalid dimensions: {video_path}")

        # Apply max_seconds limit if specified
        max_seconds = self.config.get("max_seconds")
        if max_seconds is not None:
            max_frames = int(max_seconds * info["fps"])
            if max_frames < info["total_frames"]:
                self.logger.info(
                    f"Limiting processing to {max_seconds}s ({max_frames} frames) "
                    f"instead of full video ({info['total_frames']} frames)"
                )
                info["total_frames"] = max_frames

        info["duration"] = info["total_frames"] / info["fps"]
        cap.release()
        return info

    def _stage1_calibrate_court(self, video_path: str) -> CourtCalibrationResult:
        """Stage 1: Court calibration from first frame."""
        self.logger.info("Stage 1: Court Calibration")
        stage_start = time.time()

        if self.court_calibrator is None:
            self.logger.warning("Court calibration disabled, skipping")
            return None

        # Read first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame_img = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read first frame from {video_path}")

        frame = Frame(image=frame_img, frame_number=0, timestamp=0.0)

        # Calibrate court
        calib_input = CourtCalibrationInput(frame=frame)
        calibration_result = self.court_calibrator.process_frame(calib_input)

        stage_time = time.time() - stage_start
        self.logger.info(f"Stage 1 completed in {stage_time:.2f}s")

        return calibration_result

    def _stage2_track_frames(
        self,
        video_path: str,
        calibration_result: CourtCalibrationResult,
        video_info: Dict,
    ) -> Tuple[Dict[int, List[PlayerDetectionResult]], List[BallDetectionResult]]:
        """Stage 2: Frame-by-frame tracking of players and ball."""
        self.logger.info("Stage 2: Frame-by-Frame Tracking")
        stage_start = time.time()

        cap = cv2.VideoCapture(video_path)
        total_frames = video_info["total_frames"]
        fps = video_info["fps"]

        # Storage for tracking results per frame
        player_tracking_results = {1: [], 2: []}
        ball_detection_results = []

        # Process frames
        frame_number = 0
        pbar = tqdm(
            total=total_frames,
            desc="Tracking frames",
            disable=not self.config["logging"]["show_progress"],
        )

        while cap.isOpened() and frame_number < total_frames:
            ret, frame_img = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps
            frame = Frame(
                image=frame_img, frame_number=frame_number, timestamp=timestamp
            )

            # Track players
            if self.player_tracker:
                player_input = PlayerTrackingInput(
                    frame=frame,
                    floor_homography=calibration_result.homographies.get("floor"),
                    wall_homography=calibration_result.homographies.get("wall"),
                )
                player_result = self.player_tracker.process_frame(player_input)
                player_tracking_results[1].append(player_result.get_player(1))
                player_tracking_results[2].append(player_result.get_player(2))
            else:
                player_tracking_results[1].append(
                    PlayerTrackingResult.no_players_detected(frame_number)
                )
                player_tracking_results[2].append(
                    PlayerTrackingResult.no_players_detected(frame_number)
                )
            # Track ball
            if self.ball_tracker:
                self.ball_tracker.set_is_black_ball(calibration_result.is_wall_white())
                ball_input = BallTrackingInput(frame=frame)
                ball_result = self.ball_tracker.process_frame(ball_input)
                ball_detection_results.append(ball_result)
            else:
                ball_detection_results.append(
                    BallDetectionResult.not_detected(frame_number)
                )

            frame_number += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        stage_time = time.time() - stage_start
        self.logger.info(f"Stage 2 completed in {stage_time:.2f}s")

        return player_tracking_results, ball_detection_results

    def _stage3_postprocess(
        self,
        raw_player_data: Dict[int, List[Optional[PlayerDetectionResult]]],
        raw_ball_data: List[BallDetectionResult],
    ) -> Tuple[Dict[int, PlayerTrajectory], BallTrajectory]:
        """Stage 3: Trajectory postprocessing (smoothing, interpolation)."""
        self.logger.info("Stage 3: Trajectory Postprocessing")
        stage_start = time.time()

        player_trajectories = {}
        ball_trajectory = None

        # Postprocess player trajectories using the data model
        if self.player_tracker and raw_player_data:
            postproc_input = PlayerPostprocessingInput(
                players_detections=raw_player_data
            )
            postproc_result = self.player_tracker.postprocess(postproc_input)
            player_trajectories = postproc_result.trajectories

        # Postprocess ball trajectory using the data model
        if self.ball_tracker and raw_ball_data:
            positions = [result.position for result in raw_ball_data]

            postproc_input = BallPostprocessingInput(
                positions=positions,
            )
            ball_trajectory = self.ball_tracker.postprocess(postproc_input)

        stage_time = time.time() - stage_start
        self.logger.info(f"Stage 3 completed in {stage_time:.2f}s")

        return player_trajectories, ball_trajectory

    def _stage4_segment_rallies(
        self,
        ball_trajectory: BallTrajectory,
        player_trajectories: Dict[int, PlayerTrajectory],
        video_info: Dict,
    ) -> List[RallySegment]:
        """Stage 4: Rally segmentation."""
        self.logger.info("Stage 4: Rally Segmentation")
        stage_start = time.time()

        if self.rally_detector is None or ball_trajectory is None:
            self.logger.warning(
                "Rally detection disabled or no ball trajectory, skipping"
            )
            return []

        # Extract y-coordinates from ball positions (use 0.0 for None positions)
        ball_y_positions = [
            pos.y if pos is not None else 0.0 for pos in ball_trajectory.positions
        ]

        rally_input = RallySegmentationInput(
            ball_positions=ball_y_positions,
            frame_numbers=list(range(video_info["total_frames"])),
            player_1_positions=player_trajectories.get(1).real_positions,
            player_2_positions=player_trajectories.get(2).real_positions,
        )

        # Segment rallies
        rally_result = self.rally_detector.segment_rallies(rally_input)
        rally_segments = rally_result.segments

        stage_time = time.time() - stage_start
        self.logger.info(
            f"Stage 4 completed in {stage_time:.2f}s - Found {len(rally_segments)} rallies"
        )

        return rally_segments

    def _stage5_detect_hits(
        self,
        ball_trajectory: BallTrajectory,
        player_trajectories: Dict[int, PlayerTrajectory],
        rally_segments: List[RallySegment],
    ) -> Tuple[WallHitDetectionResult, RacketHitDetectionResult]:
        """Stage 5: Wall and racket hit detection."""
        self.logger.info("Stage 5: Hit Detection")
        stage_start = time.time()

        all_wall_hits = None
        all_racket_hits = None

        if ball_trajectory is None:
            self.logger.warning("No ball trajectory, skipping hit detection")
            return all_wall_hits, all_racket_hits

        # If no rallies, process entire video as one segment
        if not rally_segments:
            rally_segments = [
                RallySegment(
                    rally_id=1,
                    start_frame=0,
                    end_frame=len(ball_trajectory.positions) - 1,
                    duration=(len(ball_trajectory.positions) - 1) / 30.0,  # Approximate
                )
            ]

        # Detect hits per rally
        for rally in tqdm(
            rally_segments,
            desc="Detecting hits",
            disable=not self.config["logging"]["show_progress"],
        ):
            # Wall hits
            if self.wall_hit_detector:
                wall_input = WallHitInput(
                    positions=ball_trajectory.positions[
                        rally.start_frame : rally.end_frame + 1
                    ],
                )
                wall_result = self.wall_hit_detector.detect(wall_input)
                all_wall_hits = wall_result

            # Racket hits
            if self.racket_hit_detector:
                racket_input = RacketHitInput(
                    positions=ball_trajectory.positions[
                        rally.start_frame : rally.end_frame + 1
                    ],
                    wall_hits=wall_result.wall_hits,
                    player_positions={
                        1: player_trajectories.get(1).positions[
                            rally.start_frame : rally.end_frame + 1
                        ],
                        2: player_trajectories.get(2).positions[
                            rally.start_frame : rally.end_frame + 1
                        ],
                    },
                )
                racket_result = self.racket_hit_detector.detect(racket_input)
                all_racket_hits = racket_result

        stage_time = time.time() - stage_start
        self.logger.info(
            f"Stage 5 completed in {stage_time:.2f}s - "
            f"Wall hits: {len(all_wall_hits.wall_hits)}, Racket hits: {len(all_racket_hits.racket_hits)}"
        )

        return all_wall_hits, all_racket_hits

    def _stage6_classify(
        self,
        player_trajectories: Dict[int, PlayerTrajectory],
        racket_hits: List[RacketHit],
        wall_hits: List[WallHit],
        video_info: Dict,
    ) -> Tuple[StrokeDetectionResult, ShotClassificationResult]:
        """Stage 6: Stroke and shot classification."""
        self.logger.info("Stage 6: Stroke and Shot Classification")
        stage_start = time.time()

        stroke_results = None
        shot_results = None

        # Stroke detection
        if self.stroke_detector and racket_hits:
            stroke_input = StrokeDetectionInput(
                player_keypoints={
                    1: player_trajectories.get(1).get_keypoints_array(),
                    2: player_trajectories.get(2).get_keypoints_array(),
                },
                racket_hits=[hit.frame for hit in racket_hits],
                racket_hit_player_ids=[hit.player_id for hit in racket_hits],
                frame_numbers=list(range(video_info["total_frames"])),
            )
            stroke_detection_result = self.stroke_detector.detect(stroke_input)
            stroke_results = stroke_detection_result

        # Shot classification
        if self.shot_classifier and racket_hits and wall_hits:
            shot_input = ShotClassificationInput(
                player1_positions_meter=player_trajectories.get(1).real_positions,
                player2_positions_meter=player_trajectories.get(2).real_positions,
                wall_hits=wall_hits,
                racket_hits=racket_hits,
            )
            shot_classification_result = self.shot_classifier.classify(shot_input)
            shot_results = shot_classification_result

        stage_time = time.time() - stage_start
        self.logger.info(
            f"Stage 6 completed in {stage_time:.2f}s - "
            f"Strokes: {len(stroke_results.strokes)}, Shots: {len(shot_results.shots)}"
        )

        return stroke_results, shot_results

    def _stage7_export(
        self,
        video_path: str,
        video_name: str,
        video_info: Dict,
        output_dir: Path,
        calibration_result: CourtCalibrationResult,
        player_trajectories: Dict[int, PlayerTrajectory],
        ball_trajectory: BallTrajectory,
        rally_segments: List[RallySegment],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
        stroke_results: List[StrokeResult],
        shot_results: List[ShotResult],
    ) -> Dict[str, str]:
        """Stage 7: Export CSV, annotated video, and statistics."""
        self.logger.info("Stage 7: Export and Visualization")
        stage_start = time.time()

        output_paths = {}

        # Build DataFrame
        df = self._build_dataframe(
            video_info=video_info,
            player_trajectories=player_trajectories,
            ball_trajectory=ball_trajectory,
            rally_segments=rally_segments,
            wall_hits=wall_hits,
            racket_hits=racket_hits,
            stroke_results=stroke_results,
            shot_results=shot_results,
        )

        # Export CSV
        if self.config["output"]["save_csv"]:
            csv_filename = self.config["output"]["csv_filename"].format(
                video_name=video_name
            )
            csv_path = output_dir / csv_filename
            df.to_csv(csv_path, index=False)
            output_paths["csv"] = str(csv_path)
            self.logger.info(f"CSV exported: {csv_path}")

        # Export statistics JSON
        if self.config["output"]["save_statistics"]:
            stats = self._compute_statistics(
                video_name=video_name,
                video_info=video_info,
                rally_segments=rally_segments,
                stroke_results=stroke_results,
                shot_results=shot_results,
                wall_hits=wall_hits,
                racket_hits=racket_hits,
            )
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
            self._render_annotated_video(
                video_path=video_path,
                output_path=video_output_path,
                df=df,
                rally_segments=rally_segments,
            )
            output_paths["video"] = str(video_output_path)
            self.logger.info(f"Annotated video exported: {video_output_path}")

        stage_time = time.time() - stage_start
        self.logger.info(f"Stage 7 completed in {stage_time:.2f}s")

        return output_paths

    def _build_dataframe(
        self,
        video_info: Dict,
        player_trajectories: Dict[int, PlayerTrajectory],
        ball_trajectory: BallTrajectory,
        rally_segments: List[RallySegment],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
        stroke_results: List[StrokeResult],
        shot_results: List[ShotResult],
    ) -> pd.DataFrame:
        """Build comprehensive DataFrame with results only for rally segments."""
        fps = video_info["fps"]

        # Get all frames that belong to rally segments
        rally_frames = []
        for rally in rally_segments:
            rally_frames.extend(range(rally.start_frame, rally.end_frame + 1))

        # If no rally segments, return empty DataFrame
        if not rally_frames:
            self.logger.warning("No rally segments found, returning empty DataFrame")
            return pd.DataFrame()

        # Sort frames to maintain order
        rally_frames = sorted(rally_frames)

        # Initialize data dictionary with only rally frames
        data = {
            "frame": rally_frames,
            "timestamp": [f / fps for f in rally_frames],
        }

        # Create a mapping from frame number to index in rally_frames list
        frame_to_idx = {frame: idx for idx, frame in enumerate(rally_frames)}
        num_rally_frames = len(rally_frames)

        # Add player data (only for rally frames)
        for player_id, traj in player_trajectories.items():
            prefix = f"player_{player_id}"
            data[f"{prefix}_x_pixel"] = [
                (
                    traj.positions[frame].x
                    if frame < len(traj.positions) and traj.positions[frame]
                    else None
                )
                for frame in rally_frames
            ]
            data[f"{prefix}_y_pixel"] = [
                (
                    traj.positions[frame].y
                    if frame < len(traj.positions) and traj.positions[frame]
                    else None
                )
                for frame in rally_frames
            ]
            data[f"{prefix}_x_meter"] = [
                (
                    traj.real_positions[frame].x
                    if frame < len(traj.real_positions) and traj.real_positions[frame]
                    else None
                )
                for frame in rally_frames
            ]
            data[f"{prefix}_y_meter"] = [
                (
                    traj.real_positions[frame].y
                    if frame < len(traj.real_positions) and traj.real_positions[frame]
                    else None
                )
                for frame in rally_frames
            ]

            # Add keypoints (initialize with None for all rally frames)
            # COCO keypoint indices: 5-16 (shoulders, elbows, wrists, hips, knees, ankles)
            keypoint_mapping = {
                "left_shoulder": 5,
                "right_shoulder": 6,
                "left_elbow": 7,
                "right_elbow": 8,
                "left_wrist": 9,
                "right_wrist": 10,
                "left_hip": 11,
                "right_hip": 12,
                "left_knee": 13,
                "right_knee": 14,
                "left_ankle": 15,
                "right_ankle": 16,
            }

            for kp_name in keypoint_mapping.keys():
                data[f"{prefix}_kp_{kp_name}_x"] = [None] * num_rally_frames
                data[f"{prefix}_kp_{kp_name}_y"] = [None] * num_rally_frames

            # Fill keypoints only for rally frames
            for frame_num in rally_frames:
                if frame_num < len(traj.keypoints):
                    kp_data = traj.keypoints[frame_num]
                    if kp_data and kp_data.xy:
                        rally_idx = frame_to_idx[frame_num]
                        for kp_name, coco_idx in keypoint_mapping.items():
                            point = kp_data.get_keypoint_as_point(coco_idx)
                            if point:
                                data[f"{prefix}_kp_{kp_name}_x"][rally_idx] = point.x
                                data[f"{prefix}_kp_{kp_name}_y"][rally_idx] = point.y

        # Add ball data (only for rally frames)
        if ball_trajectory:
            data["ball_x_pixel"] = [
                (
                    ball_trajectory.positions[frame].x
                    if frame < len(ball_trajectory.positions)
                    and ball_trajectory.positions[frame]
                    else None
                )
                for frame in rally_frames
            ]
            data["ball_y_pixel"] = [
                (
                    ball_trajectory.positions[frame].y
                    if frame < len(ball_trajectory.positions)
                    and ball_trajectory.positions[frame]
                    else None
                )
                for frame in rally_frames
            ]

        # Add rally data (map each rally frame to its rally_id)
        data["rally_id"] = [None] * num_rally_frames
        data["rally_state"] = [None] * num_rally_frames
        for rally in rally_segments:
            for frame in range(rally.start_frame, rally.end_frame + 1):
                if frame in frame_to_idx:
                    rally_idx = frame_to_idx[frame]
                    data["rally_id"][rally_idx] = rally.rally_id
                    data["rally_state"][rally_idx] = "PLAY"

        # Add hit events (only for rally frames)
        data["is_wall_hit"] = [0] * num_rally_frames
        data["wall_hit_x"] = [None] * num_rally_frames
        data["wall_hit_y"] = [None] * num_rally_frames
        for hit in wall_hits:
            if hit.frame in frame_to_idx:
                rally_idx = frame_to_idx[hit.frame]
                data["is_wall_hit"][rally_idx] = 1
                data["wall_hit_x"][rally_idx] = hit.position.x if hit.position else None
                data["wall_hit_y"][rally_idx] = hit.position.y if hit.position else None

        data["is_racket_hit"] = [0] * num_rally_frames
        data["racket_hit_player_id"] = [None] * num_rally_frames
        data["racket_hit_x"] = [None] * num_rally_frames
        data["racket_hit_y"] = [None] * num_rally_frames
        for hit in racket_hits:
            if hit.frame in frame_to_idx:
                rally_idx = frame_to_idx[hit.frame]
                data["is_racket_hit"][rally_idx] = 1
                data["racket_hit_player_id"][rally_idx] = hit.player_id
                data["racket_hit_x"][rally_idx] = (
                    hit.position.x if hit.position else None
                )
                data["racket_hit_y"][rally_idx] = (
                    hit.position.y if hit.position else None
                )

        # Add stroke data (only for rally frames)
        data["stroke_type"] = [None] * num_rally_frames
        data["stroke_confidence"] = [None] * num_rally_frames
        for stroke in stroke_results:
            if stroke.frame in frame_to_idx:
                rally_idx = frame_to_idx[stroke.frame]
                data["stroke_type"][rally_idx] = stroke.stroke_type.value
                data["stroke_confidence"][rally_idx] = stroke.confidence

        # Add shot data (only for rally frames)
        data["shot_type"] = [None] * num_rally_frames
        data["shot_direction"] = [None] * num_rally_frames
        data["shot_depth"] = [None] * num_rally_frames
        data["shot_confidence"] = [None] * num_rally_frames
        for shot in shot_results:
            if shot.frame in frame_to_idx:
                rally_idx = frame_to_idx[shot.frame]
                data["shot_type"][rally_idx] = shot.shot_type.value
                data["shot_direction"][rally_idx] = shot.direction.value
                data["shot_depth"][rally_idx] = shot.depth.value
                data["shot_confidence"][rally_idx] = shot.confidence

        return pd.DataFrame(data)

    def _compute_statistics(
        self,
        video_name: str,
        video_info: Dict,
        rally_segments: List[RallySegment],
        stroke_results: List[StrokeResult],
        shot_results: List[ShotResult],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
    ) -> Dict:
        """
        Compute high-level statistics for rally segments only.

        Note: All statistics (hits, strokes, shots) are computed only from rally segments.
        Video info includes total video metadata for context.
        """
        stats = {
            "video_info": {
                "filename": video_name,
                "duration_seconds": video_info["duration"],
                "fps": video_info["fps"],
                "total_frames": video_info["total_frames"],
                "resolution": [video_info["width"], video_info["height"]],
            },
            "rallies": {
                "total_rallies": len(rally_segments),
                "avg_duration_seconds": (
                    np.mean(
                        [r.duration_frames / video_info["fps"] for r in rally_segments]
                    )
                    if rally_segments
                    else 0
                ),
                "total_play_time_seconds": sum(
                    r.duration_frames / video_info["fps"] for r in rally_segments
                ),
            },
            "shots": {
                "total_shots": len(shot_results),
                "by_type": {},
                "by_direction": {},
                "by_depth": {},
            },
            "strokes": {},
            "hits": {
                "wall_hits": len(wall_hits),
                "racket_hits": len(racket_hits),
            },
        }

        # Rally statistics
        if rally_segments:
            longest = max(
                rally_segments, key=lambda r: r.duration_frames / video_info["fps"]
            )
            shortest = min(
                rally_segments, key=lambda r: r.duration_frames / video_info["fps"]
            )
            stats["rallies"]["longest_rally"] = {
                "rally_id": longest.rally_id,
                "duration": longest.duration_frames / video_info["fps"],
            }
            stats["rallies"]["shortest_rally"] = {
                "rally_id": shortest.rally_id,
                "duration": shortest.duration_frames / video_info["fps"],
            }

        # Shot statistics
        from collections import Counter

        if shot_results:
            shot_type_counts = Counter(shot.shot_type.value for shot in shot_results)
            direction_counts = Counter(shot.direction.value for shot in shot_results)
            depth_counts = Counter(shot.depth.value for shot in shot_results)

            stats["shots"]["by_type"] = dict(shot_type_counts)
            stats["shots"]["by_direction"] = dict(direction_counts)
            stats["shots"]["by_depth"] = dict(depth_counts)

        # Stroke statistics per player
        if stroke_results:
            stroke_by_player = {}
            for stroke in stroke_results:
                player_id = stroke.player_id
                if player_id not in stroke_by_player:
                    stroke_by_player[player_id] = []
                stroke_by_player[player_id].append(stroke.stroke_type.value)

            for player_id, strokes in stroke_by_player.items():
                stroke_counts = Counter(strokes)
                total = len(strokes)
                stats["strokes"][f"player_{player_id}"] = {
                    "total": total,
                    "forehand": stroke_counts.get("forehand", 0),
                    "backhand": stroke_counts.get("backhand", 0),
                    "forehand_pct": (
                        stroke_counts.get("forehand", 0) / total * 100
                        if total > 0
                        else 0
                    ),
                }

        return stats

    def _render_annotated_video(
        self,
        video_path: str,
        output_path: Path,
        df: pd.DataFrame,
        rally_segments: List[RallySegment],
    ):
        """Render annotated video with visualizations (only rally segments)."""
        # If no rally segments or empty DataFrame, skip video creation
        if not rally_segments or df.empty:
            self.logger.warning("No rally segments to render, skipping video creation")
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(
            *self.config["processing"]["video_writer_codec"]
        )
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Build a set of rally frames for quick lookup
        rally_frame_set = set(df["frame"].tolist())

        # Create a mapping from frame number to DataFrame row index
        frame_to_df_idx = {frame_num: idx for idx, frame_num in enumerate(df["frame"])}

        # Calculate total frames to process
        total_rally_frames = len(rally_frame_set)

        frame_number = 0
        frames_written = 0
        pbar = tqdm(
            total=total_rally_frames,
            desc="Rendering rally video",
            disable=not self.config["logging"]["show_progress"],
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process and write frames that are in rally segments
            if frame_number in rally_frame_set:
                # Get frame data from DataFrame
                df_idx = frame_to_df_idx[frame_number]
                frame_data = df.iloc[df_idx]

                # Draw visualizations
                frame = self._draw_frame_annotations(
                    frame,
                    frame_data,
                )

                out.write(frame)
                frames_written += 1
                pbar.update(1)

            frame_number += 1

        pbar.close()
        cap.release()
        out.release()

        self.logger.info(
            f"Rendered {frames_written} frames from {len(rally_segments)} rally segments"
        )

    def _draw_frame_annotations(
        self,
        frame: np.ndarray,
        frame_data: pd.Series,
    ) -> np.ndarray:
        """Draw annotations on a single frame."""
        vis_config = self.config["visualization"]

        # Draw player boxes and keypoints
        if vis_config["draw_player_boxes"] or vis_config["draw_player_keypoints"]:
            for player_id in [1, 2]:
                x = frame_data.get(f"player_{player_id}_x_pixel")
                y = frame_data.get(f"player_{player_id}_y_pixel")

                if pd.notna(x) and pd.notna(y):
                    x, y = int(x), int(y)

                    # Draw position circle
                    cv2.circle(
                        frame, (x, y), 10, tuple(vis_config["player_box_color"]), -1
                    )
                    # Draw player ID
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
            ball_x = frame_data.get("ball_x_pixel")
            ball_y = frame_data.get("ball_y_pixel")
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
            if pd.notna(rally_id):
                cv2.putText(
                    frame,
                    f"Rally {int(rally_id)}",
                    tuple(vis_config["rally_id_position"]),
                    vis_config["font"],
                    vis_config["rally_id_font_scale"],
                    tuple(vis_config["rally_id_color"]),
                    vis_config["font_thickness"],
                )

        # Draw stroke labels
        if vis_config["show_stroke_labels"]:
            stroke_type = frame_data.get("stroke_type")
            if pd.notna(stroke_type):
                cv2.putText(
                    frame,
                    f"Stroke: {stroke_type}",
                    (20, 100),
                    vis_config["font"],
                    vis_config["stroke_label_font_scale"],
                    tuple(vis_config["stroke_label_color"]),
                    vis_config["font_thickness"],
                )

        # Draw shot labels
        if vis_config["show_shot_labels"]:
            shot_type = frame_data.get("shot_type")
            if pd.notna(shot_type):
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
