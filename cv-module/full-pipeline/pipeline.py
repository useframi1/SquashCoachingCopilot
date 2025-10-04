import cv2
import numpy as np
from court_detection_pipeline import CourtCalibrator
from player_tracking_pipeline import PlayerTracker
from rally_state_pipeline import RallyStateDetector
from ball_detection_pipeline import BallTracker
from utils import get_real_coordinates


class Pipeline:
    """Integrated pipeline for squash video analysis."""

    def __init__(self):
        """Initialize all four pipelines."""
        self.court_calibrator = CourtCalibrator()
        self.player_tracker = None
        self.rally_state_detector = RallyStateDetector()
        self.ball_tracker = BallTracker()

        # Store results
        self.homographies = None
        self.keypoints = None
        self.is_calibrated = False

    def process_video(
        self, video_path: str, output_path: str = None, display: bool = True
    ):
        """
        Process video through all four pipelines and display results.

        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            display: Whether to display video in real-time
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame through all pipelines
            annotated_frame = self.process_frame(frame)

            # Write to output video if specified
            if writer:
                writer.write(annotated_frame)

            # Display frame if requested
            if display:
                cv2.imshow("Squash Analysis Pipeline", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Processing interrupted by user")
                    break

            frame_count += 1

            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        print(f"Processing complete. Processed {frame_count} frames.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through all pipelines.

        Args:
            frame: Input frame

        Returns:
            Annotated frame with all pipeline results
        """
        annotated_frame = frame.copy()

        # 1. Court Detection (calibrate on first frame)
        if not self.is_calibrated:
            self.homographies, self.keypoints = self.court_calibrator.process_frame(
                frame
            )
            self.player_tracker = PlayerTracker(
                homography=self.court_calibrator.get_homography("t-boxes")
            )
            self.is_calibrated = True
            annotated_frame = self._draw_court_keypoints(
                annotated_frame, self.keypoints
            )

        # 2. Player Tracking
        player_results = self.player_tracker.process_frame(frame)
        annotated_frame = self._draw_player_tracking(annotated_frame, player_results)

        # 3. Ball Detection
        ball_x, ball_y = self.ball_tracker.process_frame(frame)
        annotated_frame = self._draw_ball_detection(annotated_frame, ball_x, ball_y)

        # 4. Rally State Detection (requires real court coordinates)
        real_coordinates = get_real_coordinates(player_results)
        rally_state = self.rally_state_detector.process_frame(real_coordinates)
        annotated_frame = self._draw_rally_state(annotated_frame, rally_state)

        return annotated_frame

    def _draw_court_keypoints(self, frame: np.ndarray, keypoints: dict) -> np.ndarray:
        """Draw court keypoints on frame."""
        if keypoints is None:
            return frame

        for class_name, kp_array in keypoints.items():
            for i, (x, y) in enumerate(kp_array):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        return frame

    def _draw_player_tracking(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw player tracking results on frame."""
        for player_id in [1, 2]:
            if results[player_id]["bbox"]:
                x1, y1, x2, y2 = map(int, results[player_id]["bbox"])
                color = (0, 255, 0) if player_id == 1 else (255, 0, 0)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw position
                pos = results[player_id]["position"]
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, color, -1)

                # Draw label
                label = f"P{player_id}: {results[player_id]['confidence']:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        return frame

    def _draw_ball_detection(self, frame: np.ndarray, ball_x, ball_y) -> np.ndarray:
        """Draw ball detection results on frame."""
        if ball_x is not None and ball_y is not None:
            cv2.circle(frame, (ball_x, ball_y), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "Ball",
                (ball_x + 10, ball_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        return frame

    def _draw_rally_state(self, frame: np.ndarray, rally_state: str) -> np.ndarray:
        """Draw rally state on frame."""
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
