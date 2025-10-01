import json
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from config import CONFIG
from player_tracker import PlayerTracker


class PlayerTrackerEvaluator:
    def __init__(self):
        self.coco_json_path = CONFIG["paths"]["coco_annotations"]
        self.iou_threshold = CONFIG["evaluation"]["iou_threshold"]
        self.ground_truth = self.load_coco_annotations()

        # Initialize tracker
        self.tracker = PlayerTracker()

        # Tracking evaluation metrics
        self.frame_results = []
        self.id_mapping = None
        self.mapping_confidence = 0.0

        # Accumulated metrics
        self.total_gt_boxes = 0
        self.total_pred_boxes = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.id_switches = 0
        self.track_breaks = 0

        # Per-player metrics
        player_1_id = CONFIG["evaluation"]["player_1_class_id"]
        player_2_id = CONFIG["evaluation"]["player_2_class_id"]
        self.player_metrics = {
            player_1_id: {"tp": 0, "fp": 0, "fn": 0},
            player_2_id: {"tp": 0, "fp": 0, "fn": 0},
        }

    def load_coco_annotations(self):
        """Load and organize COCO annotations by frame"""
        with open(self.coco_json_path, "r") as f:
            coco_data = json.load(f)

        print(f"Loading COCO annotations...")
        print(f"Categories: {coco_data['categories']}")

        images = {img["id"]: img for img in coco_data["images"]}
        annotations_by_frame = defaultdict(list)

        for ann in coco_data["annotations"]:
            annotations_by_frame[ann["image_id"]].append(ann)

        ground_truth = {}
        for img_id, anns in annotations_by_frame.items():
            frame_name = images[img_id]["extra"]["name"]
            ground_truth[frame_name] = []

            for ann in anns:
                x, y, w, h = ann["bbox"]
                bbox = [x, y, x + w, y + h]
                ground_truth[frame_name].append(
                    {
                        "bbox": bbox,
                        "class_id": ann["category_id"],
                        "annotation_id": ann["id"],
                    }
                )

        print(f"Loaded {len(ground_truth)} frames with annotations")
        return ground_truth

    def frame_name_formatter(self, frame_number):
        """Format frame name according to config"""
        pattern = CONFIG["frame_formatting"]["pattern"]
        video_name = CONFIG["frame_formatting"]["video_name"]
        return pattern.format(video_name=video_name, frame_number=frame_number)

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def establish_id_mapping(self, predictions, ground_truth_frame, frame_name):
        """Establish or verify the mapping between tracker IDs and ground truth classes"""
        if len(predictions) == 0 or len(ground_truth_frame) == 0:
            return

        iou_matrix = np.zeros((len(predictions), len(ground_truth_frame)))
        for i, pred in enumerate(predictions):
            pred_bbox = pred["bbox"]
            for j, gt in enumerate(ground_truth_frame):
                gt_bbox = gt["bbox"]
                iou_matrix[i, j] = self.calculate_iou(pred_bbox, gt_bbox)

        if iou_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)

            potential_mapping = {}
            mapping_ious = []

            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] > self.iou_threshold:
                    tracker_id = predictions[row]["tracker_id"]
                    gt_class = ground_truth_frame[col]["class_id"]
                    potential_mapping[tracker_id] = gt_class
                    mapping_ious.append(iou_matrix[row, col])

            current_confidence = np.mean(mapping_ious) if mapping_ious else 0
            if self.id_mapping is None or current_confidence > self.mapping_confidence:
                self.id_mapping = potential_mapping.copy()
                self.mapping_confidence = current_confidence
                print(
                    f"Updated ID mapping at {frame_name}: {self.id_mapping} "
                    f"(confidence: {current_confidence:.3f})"
                )

    def evaluate_frame(self, predictions, frame_name):
        """Evaluate a single frame against ground truth"""
        if frame_name not in self.ground_truth:
            return

        gt_frame = self.ground_truth[frame_name]
        self.total_gt_boxes += len(gt_frame)
        self.total_pred_boxes += len(predictions)

        self.establish_id_mapping(predictions, gt_frame, frame_name)

        if self.id_mapping is None:
            return

        mapped_predictions = []
        for pred in predictions:
            tracker_id = pred["tracker_id"]
            if tracker_id in self.id_mapping:
                mapped_pred = pred.copy()
                mapped_pred["class_id"] = self.id_mapping[tracker_id]
                mapped_predictions.append(mapped_pred)

        iou_matrix = np.zeros((len(mapped_predictions), len(gt_frame)))
        for i, pred in enumerate(mapped_predictions):
            for j, gt in enumerate(gt_frame):
                iou_matrix[i, j] = self.calculate_iou(pred["bbox"], gt["bbox"])

        matched_pred = set()
        matched_gt = set()

        if iou_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)

            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] > self.iou_threshold:
                    pred = mapped_predictions[row]
                    gt = gt_frame[col]

                    if pred["class_id"] == gt["class_id"]:
                        self.true_positives += 1
                        self.player_metrics[gt["class_id"]]["tp"] += 1
                    else:
                        self.false_positives += 1
                        self.false_negatives += 1
                        self.id_switches += 1
                        self.player_metrics[pred["class_id"]]["fp"] += 1
                        self.player_metrics[gt["class_id"]]["fn"] += 1

                    matched_pred.add(row)
                    matched_gt.add(col)

        for i in range(len(mapped_predictions)):
            if i not in matched_pred:
                self.false_positives += 1
                pred_class = mapped_predictions[i]["class_id"]
                if pred_class in self.player_metrics:
                    self.player_metrics[pred_class]["fp"] += 1

        for j in range(len(gt_frame)):
            if j not in matched_gt:
                self.false_negatives += 1
                gt_class = gt_frame[j]["class_id"]
                if gt_class in self.player_metrics:
                    self.player_metrics[gt_class]["fn"] += 1

        self.frame_results.append(
            {
                "frame_name": frame_name,
                "predictions": len(predictions),
                "ground_truth": len(gt_frame),
                "true_positives": len(matched_pred) if matched_pred else 0,
                "id_mapping_used": self.id_mapping.copy() if self.id_mapping else None,
            }
        )

    def run_video(self):
        """Run tracker on video and evaluate against ground truth"""
        cap = cv2.VideoCapture(CONFIG["paths"]["test_video"])
        frame_count = 0

        video_writer = None
        if CONFIG["paths"]["output_video"]:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            codec = CONFIG["output"]["video_codec"]
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(
                CONFIG["paths"]["output_video"], fourcc, fps, (width, height)
            )
            print(f"Output video will be saved to: {CONFIG['paths']['output_video']}")

        print(f"Processing video: {CONFIG['paths']['test_video']}")

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
            if frame_name in self.ground_truth:
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

                self.evaluate_frame(predictions, frame_name)

            # Draw tracking visualization
            if CONFIG["visualization"]["display"] or video_writer:
                for player_id in [1, 2]:
                    if results[player_id]["position"]:
                        color_key = f"player_{player_id}_color"
                        color = tuple(CONFIG["visualization"][color_key])
                        pos = results[player_id]["position"]
                        radius = CONFIG["visualization"]["circle_radius"]
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), radius, color, -1)

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

            if video_writer:
                video_writer.write(frame)

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
            print(f"Output video saved to: {CONFIG['paths']['output_video']}")
        if CONFIG["visualization"]["display"]:
            cv2.destroyAllWindows()

        print(f"Completed: {frame_count} frames processed")

        metrics_results = self.calculate_final_metrics()

        if CONFIG["paths"]["output_results"]:
            self.save_results_to_txt(metrics_results)

        return metrics_results

    def calculate_final_metrics(self):
        """Calculate final tracking metrics"""
        if self.total_gt_boxes == 0:
            return {}

        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else 0
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        mota = (
            1
            - (self.false_negatives + self.false_positives + self.id_switches)
            / self.total_gt_boxes
            if self.total_gt_boxes > 0
            else 0
        )

        player_stats = {}
        player_1_id = CONFIG["evaluation"]["player_1_class_id"]
        player_2_id = CONFIG["evaluation"]["player_2_class_id"]

        for player_id in [player_1_id, player_2_id]:
            if player_id in self.player_metrics:
                metrics = self.player_metrics[player_id]
                tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]

                p_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                p_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                p_f1 = (
                    2 * (p_precision * p_recall) / (p_precision + p_recall)
                    if (p_precision + p_recall) > 0
                    else 0
                )

                player_stats[f"player_{player_id}"] = {
                    "precision": p_precision,
                    "recall": p_recall,
                    "f1_score": p_f1,
                    "detections": tp + fn,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                }

        return {
            "overall": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "mota": mota,
                "total_frames": len(self.frame_results),
                "id_switches": self.id_switches,
                "id_mapping": self.id_mapping,
                "mapping_confidence": self.mapping_confidence,
            },
            "per_player": player_stats,
            "detection_stats": {
                "total_ground_truth": self.total_gt_boxes,
                "total_predictions": self.total_pred_boxes,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            },
        }

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
            results = self.calculate_final_metrics()

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
        """Run complete evaluation pipeline"""
        results = self.run_video()
        self.print_results(results)


if __name__ == "__main__":
    evaluator = PlayerTrackerEvaluator()
    evaluator.run_evaluation()
