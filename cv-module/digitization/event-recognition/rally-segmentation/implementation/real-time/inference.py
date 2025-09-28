import numpy as np
import pandas as pd
import cv2
from config import CONFIG

from modeling.predictor import StatePredictor


class VideoInferencePipeline:
    """
    Complete video inference pipeline using existing MetricsAggregator.
    """

    def __init__(self):
        """
        Initialize video inference pipeline.

        Args:
            video_path: Path to video file
            model_path: Path to trained model
        """
        self.video_path = CONFIG["inference"]["video_path"]
        self.predictor = StatePredictor()

        # Import MetricsAggregator from utilities
        from utilities.metrics_aggregator import MetricsAggregator

        self.metrics_aggregator = MetricsAggregator(window_size=CONFIG["window_size"])

        self.predictions = []
        self.base_metrics_history = []

    def process_video(self):
        """
        Process video and make state predictions.
        """
        start_frame = CONFIG["inference"]["start_frame"]
        end_frame = CONFIG["inference"]["end_frame"]
        save_output = CONFIG["inference"]["save_output"]
        output_path = CONFIG["inference"]["video_output_path"]

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if end_frame is None:
            end_frame = total_frames

        print(f"Processing video: {self.video_path}")
        print(f"Frames: {start_frame} to {end_frame}")
        print(f"Window size: {CONFIG['window_size']} frames")

        # Initialize video writer if saving output
        writer = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (frame_width, frame_height)
            )

        # Initialize court calibration and tracking
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        self.metrics_aggregator.calibrate_court(first_frame)
        self.metrics_aggregator.initialize_tracker(first_frame)

        # Reset predictor state
        self.predictor.reset_state()

        # Process frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_num = start_frame
        current_prediction = "start"

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Update metrics aggregator
            self.metrics_aggregator.update_metrics(frame, frame_num)

            # Check if we have a full window for prediction
            if self.metrics_aggregator.has_full_window():
                # Get aggregated base metrics
                base_metrics = self.metrics_aggregator.get_aggregated_metrics()

                if base_metrics and base_metrics["mean_distance"] is not None:
                    # Make prediction
                    prediction = self.predictor.predict_single(base_metrics)
                    current_prediction = prediction

                    # Store results
                    self.predictions.append(
                        {
                            "frame_number": frame_num,
                            "window_end_frame": frame_num,
                            "predicted_state": prediction,
                            **base_metrics,
                        }
                    )

                    print(
                        f"Frame {frame_num}: {prediction} (distance: {base_metrics['mean_distance']:.2f})"
                    )

            # Save frame with prediction overlay if requested
            if writer:
                annotated_frame = self._draw_prediction_overlay(
                    frame, current_prediction, frame_num
                )
                writer.write(annotated_frame)

                cv2.imshow("Rally State Prediction", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                # If not saving, still show the video
                display_frame = self._draw_prediction_overlay(
                    frame, current_prediction, frame_num
                )
                cv2.imshow("Rally State Prediction", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_num += 1

            if frame_num % 100 == 0:
                progress = (frame_num - start_frame) / (end_frame - start_frame) * 100
                print(f"Progress: {progress:.1f}%")

        cap.release()
        if writer:
            writer.release()
            print(f"Output video saved: {output_path}")

        print(f"Processing complete. Made {len(self.predictions)} predictions.")
        return self.predictions

    def _draw_prediction_overlay(
        self, frame: np.ndarray, prediction: str, frame_num: int
    ) -> np.ndarray:
        """Draw prediction overlay on frame."""
        annotated_frame = frame.copy()

        # State colors
        colors = {"start": (255, 200, 0), "active": (0, 255, 0), "end": (0, 0, 255)}
        color = colors.get(prediction, (128, 128, 128))

        # Draw banner
        height, width = frame.shape[:2]
        banner_height = 60

        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

        # Draw text
        text = f"Frame {frame_num}: {prediction.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, text, (10, 35), font, 1.2, (255, 255, 255), 2)

        # Draw player bounding boxes
        for player_id, bbox in self.metrics_aggregator.last_player_bboxes.items():
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                player_color = (0, 255, 0) if player_id == 1 else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), player_color, 2)
                cv2.putText(
                    annotated_frame,
                    f"P{player_id}",
                    (x1, y1 - 10),
                    font,
                    0.6,
                    player_color,
                    2,
                )

        return annotated_frame

    def save_predictions_csv(self):
        """Save predictions to CSV file."""
        if not self.predictions:
            print("No predictions to save")
            return

        df = pd.DataFrame(self.predictions)
        df.to_csv(CONFIG["inference"]["predictions_output_path"], index=False)
        print(f"Predictions saved to: {CONFIG['inference']['predictions_output_path']}")

    def get_predictions_dataframe(self) -> pd.DataFrame:
        """Get predictions as DataFrame."""
        return pd.DataFrame(self.predictions)

    def run_inference_pipeline(self):
        self.process_video()
        self.save_predictions_csv()


if __name__ == "__main__":
    inference_pipeline = VideoInferencePipeline()
    inference_pipeline.run_inference_pipeline()
