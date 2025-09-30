import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from squash_player_tracker import SquashPlayerTracker
from evaluator import SquashTrackerEvaluator


class TrackerEvaluationWrapper:
    """
    Wrapper class that runs the tracker and evaluates it using the evaluator.
    Keeps tracker and evaluator cleanly separated.
    """

    def __init__(self):
        """
        Args:
            tracker: SquashPlayerTracker instance
            evaluator: SquashTrackerEvaluator instance
            frame_name_formatter: Function that takes frame_number and returns frame_name
                                 Default: lambda n: f"frame_{n:04d}.jpg"
        """
        self.tracker = SquashPlayerTracker()
        self.evaluator = SquashTrackerEvaluator()

    def frame_name_formatter(self, frame_number):
        pattern = CONFIG["frame_formatting"]["pattern"]
        video_name = CONFIG["frame_formatting"]["video_name"]
        return pattern.format(video_name=video_name, frame_number=frame_number)

    def run_video(self):
        """
        Run tracker on video and evaluate against ground truth.

        Returns:
            dict: Evaluation results
        """
        cap = cv2.VideoCapture(CONFIG["paths"]["test_video"])
        frame_count = 0

        # Setup video writer if output path is provided
        video_writer = None
        if CONFIG["paths"]["output_video"]:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create video writer with MP4 codec
            codec = CONFIG["output"]["video_codec"]
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(
                CONFIG["paths"]["output_video"], fourcc, fps, (width, height)
            )
            print(f"Output video will be saved to: {CONFIG["paths"]["output_video"]}")

        print(f"Processing video: {CONFIG["paths"]["test_video"]}")

        while cap.isOpened():
            if (
                CONFIG["processing"]["max_frames"]
                and frame_count >= CONFIG["processing"]["max_frames"]
            ):
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Track frame
            results = self.tracker.track_frame(frame)

            # Get frame name for evaluation
            frame_name = self.frame_name_formatter(frame_count)

            # Evaluate if ground truth exists for this frame
            if frame_name in self.evaluator.ground_truth:
                predictions = []
                for player_id in [1, 2]:
                    if results[player_id]["bbox"] is not None:
                        predictions.append(
                            {
                                "bbox": results[player_id]["bbox"],
                                "tracker_id": player_id,
                                "confidence": results[player_id]["confidence"],
                            }
                        )

                self.evaluator.evaluate_frame(predictions, frame_name)

            # Draw tracking visualization (for both display and video output)
            if CONFIG["visualization"]["display"] or video_writer:
                for player_id in [1, 2]:
                    if results[player_id]["position"]:
                        color_key = f"player_{player_id}_color"
                        color = tuple(CONFIG["visualization"][color_key])
                        pos = results[player_id]["position"]
                        radius = CONFIG["visualization"]["circle_radius"]
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), radius, color, -1)

                        # Draw bbox
                        if results[player_id]["bbox"]:
                            bbox = results[player_id]["bbox"]
                            thickness = CONFIG["visualization"]["bbox_thickness"]
                            cv2.rectangle(
                                frame,
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
                                color,
                                thickness,
                            )
                            cv2.putText(
                                frame,
                                f"P{player_id}",
                                (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

            # Save frame to output video if writer is initialized
            if video_writer:
                video_writer.write(frame)

            # Display frame if visualization is enabled
            if CONFIG["visualization"]["display"]:
                window_name = CONFIG["visualization"]["window_name"]
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

            if frame_count % CONFIG["processing"]["progress_interval"] == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        if video_writer:
            video_writer.release()
            print(f"Output video saved to: {CONFIG["paths"]["output_video"]}")
        if CONFIG["visualization"]["display"]:
            cv2.destroyAllWindows()

        print(f"Completed: {frame_count} frames processed")

        # Calculate final metrics
        metrics_results = self.evaluator.calculate_final_metrics()

        # Save results to TXT file if path is provided
        if CONFIG["paths"]["output_results"]:
            self.save_results_to_txt(metrics_results)

        return metrics_results

    def save_results_to_txt(self, results):
        """Save evaluation results to a TXT file"""
        with open(CONFIG["paths"]["output_results"], "w") as f:
            overall = results["overall"]

            f.write("=" * 60 + "\n")
            f.write("TRACKING EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")

            f.write("OVERALL PERFORMANCE:\n")
            precision = CONFIG["output"]["results_precision"]
            f.write(f"   • Precision: {overall['precision']:.{precision}f}\n")
            f.write(f"   • Recall: {overall['recall']:.{precision}f}\n")
            f.write(f"   • F1-Score: {overall['f1_score']:.{precision}f}\n")
            f.write(f"   • MOTA: {overall['mota']:.{precision}f}\n")
            f.write(f"   • ID Switches: {overall['id_switches']}\n")
            f.write(f"   • Frames Evaluated: {overall['total_frames']}\n\n")

            if overall["id_mapping"]:
                f.write(
                    f"ID MAPPING (Confidence: {overall['mapping_confidence']:.{precision}f}):\n"
                )
                for tracker_id, gt_class in overall["id_mapping"].items():
                    f.write(f"   • Tracker {tracker_id} → GT Player {gt_class}\n")
                f.write("\n")

            f.write("PER-PLAYER PERFORMANCE:\n")
            for player_key, stats in results["per_player"].items():
                f.write(f"   {player_key}:\n")
                f.write(f"      - Precision: {stats['precision']:.{precision}f}\n")
                f.write(f"      - Recall: {stats['recall']:.{precision}f}\n")
                f.write(f"      - F1: {stats['f1_score']:.{precision}f}\n")
                f.write(f"      - GT Instances: {stats['detections']}\n")

        print(f"Results saved to: {CONFIG['paths']['output_results']}")

    def print_results(self, results=None):
        """Print formatted evaluation results"""
        if results is None:
            results = self.evaluator.calculate_final_metrics()

        overall = results["overall"]

        print("\n" + "=" * 60)
        print("TRACKING EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nOVERALL PERFORMANCE:")
        print(f"   • Precision: {overall['precision']:.3f}")
        print(f"   • Recall: {overall['recall']:.3f}")
        print(f"   • F1-Score: {overall['f1_score']:.3f}")
        print(f"   • MOTA: {overall['mota']:.3f}")
        print(f"   • ID Switches: {overall['id_switches']}")
        print(f"   • Frames Evaluated: {overall['total_frames']}")

        if overall["id_mapping"]:
            print(f"\nID MAPPING (Confidence: {overall['mapping_confidence']:.3f}):")
            for tracker_id, gt_class in overall["id_mapping"].items():
                print(f"   • Tracker {tracker_id} → GT Player {gt_class}")

        print(f"\nPER-PLAYER PERFORMANCE:")
        for player_key, stats in results["per_player"].items():
            print(f"   {player_key}:")
            print(f"      - Precision: {stats['precision']:.3f}")
            print(f"      - Recall: {stats['recall']:.3f}")
            print(f"      - F1: {stats['f1_score']:.3f}")
            print(f"      - GT Instances: {stats['detections']}")

    def run_evaluation(self):
        results = self.run_video()
        self.print_results(results)


if __name__ == "__main__":
    wrapper = TrackerEvaluationWrapper()
    wrapper.run_evaluation()
