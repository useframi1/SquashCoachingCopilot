"""
Player Tracker Evaluator

Evaluates player tracking performance against COCO-format ground truth annotations.
Uses the new DataFrame-based pipeline architecture.
"""

import json
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import os
from typing import Dict, List, Optional
import tqdm

from squashcopilot.modules.player_tracking import PlayerTracker
from squashcopilot.modules.court_calibration import CourtCalibrator
from squashcopilot.common.utils import load_config
from squashcopilot.common import (
    Frame,
    BoundingBox,
    CourtCalibrationInput,
    CourtCalibrationOutput,
    PlayerTrackingInput,
    PlayerPostprocessingInput,
    PlayerPostprocessingOutput,
)
from squashcopilot.common.models import player_tracking_outputs_to_dataframe


class PlayerTrackerEvaluator:
    def __init__(self, config: dict = None):
        if config is None:
            # Load the tests section from the player_tracking config
            full_config = load_config(config_name="player_tracking")
            config = full_config["tests"]
        self.config = config

        # Get test directory path
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

        # Resolve paths relative to test directory
        self.coco_json_path = os.path.join(
            self.test_dir, self.config["paths"]["coco_annotations"]
        )
        self.test_video_path = os.path.join(
            self.test_dir, self.config["paths"]["test_video"]
        )
        self.output_video_path = (
            os.path.join(self.test_dir, self.config["paths"]["output_video"])
            if self.config["paths"]["output_video"]
            else None
        )
        self.output_results_path = (
            os.path.join(self.test_dir, self.config["paths"]["output_results"])
            if self.config["paths"]["output_results"]
            else None
        )

        self.iou_threshold = self.config["evaluation"]["iou_threshold"]
        self.ground_truth = self.load_coco_annotations()

        # Initialize court calibrator to get homography
        print("Initializing court calibrator...")
        self.court_calibrator = CourtCalibrator()

        # Compute homography from first frame
        cap = cv2.VideoCapture(self.test_video_path)
        ret, first_frame_img = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read first frame from test video")

        # Create Frame and CourtCalibrationInput
        first_frame = Frame(image=first_frame_img, frame_number=0, timestamp=0.0)
        calibration_input = CourtCalibrationInput(frame=first_frame)
        self.calibration: CourtCalibrationOutput = self.court_calibrator.process_frame(
            calibration_input
        )

        if not self.calibration.calibration_success:
            raise ValueError("Failed to compute homography from first frame")

        print("Homography computed successfully")

        # Initialize tracker with calibration
        self.tracker = PlayerTracker()
        self.tracker.set_calibration(self.calibration)

        mask = self.tracker._create_court_mask(first_frame_img.shape)
        if mask is not None:
            # Create visualization
            mask_viz = cv2.cvtColor(first_frame_img.copy(), cv2.COLOR_BGR2BGRA)
            # Overlay mask in semi-transparent green
            mask_overlay = np.zeros_like(mask_viz)
            mask_overlay[mask > 0] = [0, 255, 0, 128]  # Green with 50% transparency

            # Blend
            mask_viz = cv2.addWeighted(mask_viz, 1.0, mask_overlay, 0.3, 0)

            # Display and wait for key press
            cv2.imshow("Court Mask Preview - Press any key to continue", mask_viz)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Warning: Could not create court mask")

    def reset_metrics(self):
        """Reset all tracking evaluation metrics"""
        self.frame_results = []
        self.id_mapping = None
        self.mapping_confidence = 0.0

        self.total_gt_boxes = 0
        self.total_pred_boxes = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.id_switches = 0
        self.track_breaks = 0

        player_1_id = self.config["evaluation"]["player_1_class_id"]
        player_2_id = self.config["evaluation"]["player_2_class_id"]
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
        pattern = self.config["frame_formatting"]["pattern"]
        video_name = self.config["frame_formatting"]["video_name"]
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

    def _establish_id_mapping(
        self,
        player_bboxes: Dict[int, List[float]],
        ground_truth_frame,
        frame_name: str,
    ):
        """Establish or verify the mapping between tracker IDs and ground truth classes.

        Args:
            player_bboxes: Dict mapping player_id to bbox [x1, y1, x2, y2]
            ground_truth_frame: Ground truth annotations for this frame
            frame_name: Frame name for logging
        """
        if len(player_bboxes) == 0 or len(ground_truth_frame) == 0:
            return

        player_ids = list(player_bboxes.keys())
        player_bbox_list = [player_bboxes[pid] for pid in player_ids]

        iou_matrix = np.zeros((len(player_ids), len(ground_truth_frame)))
        for i, bbox in enumerate(player_bbox_list):
            for j, gt in enumerate(ground_truth_frame):
                gt_bbox = gt["bbox"]
                iou_matrix[i, j] = self.calculate_iou(bbox, gt_bbox)

        if iou_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)

            potential_mapping = {}
            mapping_ious = []

            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] > self.iou_threshold:
                    tracker_id = player_ids[row]
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

    def evaluate_frame(self, player_bboxes: Dict[int, List[float]], frame_name: str):
        """Evaluate a single frame against ground truth.

        Args:
            player_bboxes: Dict mapping player_id to bbox [x1, y1, x2, y2]
            frame_name: Frame name for ground truth lookup
        """
        if frame_name not in self.ground_truth:
            return

        gt_frame = self.ground_truth[frame_name]

        self.total_gt_boxes += len(gt_frame)
        self.total_pred_boxes += len(player_bboxes)

        self._establish_id_mapping(player_bboxes, gt_frame, frame_name)

        if self.id_mapping is None:
            return

        # Filter players that are in the ID mapping
        mapped_player_ids = [
            pid for pid in player_bboxes.keys() if pid in self.id_mapping
        ]

        if len(mapped_player_ids) == 0:
            return

        iou_matrix = np.zeros((len(mapped_player_ids), len(gt_frame)))
        for i, player_id in enumerate(mapped_player_ids):
            player_bbox = player_bboxes[player_id]
            for j, gt in enumerate(gt_frame):
                iou_matrix[i, j] = self.calculate_iou(player_bbox, gt["bbox"])

        matched_pred = set()
        matched_gt = set()

        if iou_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)

            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] > self.iou_threshold:
                    player_id = mapped_player_ids[row]
                    pred_class = self.id_mapping[player_id]
                    gt = gt_frame[col]

                    if pred_class == gt["class_id"]:
                        self.true_positives += 1
                        self.player_metrics[gt["class_id"]]["tp"] += 1
                    else:
                        self.false_positives += 1
                        self.false_negatives += 1
                        self.id_switches += 1
                        self.player_metrics[pred_class]["fp"] += 1
                        self.player_metrics[gt["class_id"]]["fn"] += 1

                    matched_pred.add(row)
                    matched_gt.add(col)

        for i in range(len(mapped_player_ids)):
            if i not in matched_pred:
                self.false_positives += 1
                pred_class = self.id_mapping[mapped_player_ids[i]]
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
                "predictions": len(player_bboxes),
                "ground_truth": len(gt_frame),
                "true_positives": len(matched_pred) if matched_pred else 0,
                "id_mapping_used": self.id_mapping.copy() if self.id_mapping else None,
            }
        )

    def frame_generator(self):
        """Generate frames from video one at a time"""
        cap = cv2.VideoCapture(self.test_video_path)
        frame_count = 0

        while cap.isOpened():
            if (
                self.config["processing"]["max_frames"]
                and frame_count >= self.config["processing"]["max_frames"]
            ):
                break

            ret, frame = cap.read()
            if not ret:
                break

            yield frame
            frame_count += 1

        cap.release()

    def process_frames(
        self,
    ) -> tuple[pd.DataFrame, Dict[int, List[Optional[BoundingBox]]]]:
        """Process frames and return tracking results.

        Returns:
            Tuple of (DataFrame with tracking data, player_bboxes dict)
        """
        results_list = []

        # Get total frame count for progress bar
        cap = cv2.VideoCapture(self.test_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        max_frames = self.config["processing"]["max_frames"]
        if max_frames:
            total_frames = min(total_frames, max_frames)

        for frame_idx, frame_img in tqdm.tqdm(
            enumerate(self.frame_generator()),
            total=total_frames,
            desc="Processing frames",
        ):
            # Create Frame object
            frame = Frame(
                image=frame_img, frame_number=frame_idx, timestamp=frame_idx / 30.0
            )

            # Create PlayerTrackingInput
            tracking_input = PlayerTrackingInput(
                frame=frame,
                calibration=self.calibration,
            )

            # Track frame
            output = self.tracker.process_frame(tracking_input)
            results_list.append(output)

        print(f"Processing complete: {len(results_list)} frames")

        # Convert to DataFrame + complex data using the model function
        df, complex_data = player_tracking_outputs_to_dataframe(results_list)

        # Build player_bboxes dict from complex_data
        player_bboxes = {
            1: complex_data["player_1_bboxes"],
            2: complex_data["player_2_bboxes"],
        }

        return df, player_bboxes

    def apply_postprocessing(
        self,
        df: pd.DataFrame,
        player_bboxes: Dict[int, List[Optional[BoundingBox]]],
    ) -> PlayerPostprocessingOutput:
        """Apply postprocessing to tracking results using DataFrame-based pipeline.

        Args:
            df: DataFrame with tracking data
            player_bboxes: Dict mapping player_id to list of bboxes

        Returns:
            PlayerPostprocessingOutput with processed df, keypoints, and bboxes
        """
        print("Applying postprocessing (interpolation + smoothing)...")

        # For now, keypoints are empty - can be extended later
        player_keypoints: Dict[int, List[Optional[np.ndarray]]] = {
            1: [None] * len(player_bboxes.get(1, [])),
            2: [None] * len(player_bboxes.get(2, [])),
        }

        # Create PlayerPostprocessingInput
        postprocess_input = PlayerPostprocessingInput(
            df=df,
            player_keypoints=player_keypoints,
            player_bboxes=player_bboxes,
        )

        # Apply postprocessing - returns PlayerPostprocessingOutput
        postprocess_output = self.tracker.postprocess(postprocess_input)

        print(
            f"Postprocessing complete. Gaps filled: "
            f"P1={postprocess_output.num_player_1_gaps_filled}, "
            f"P2={postprocess_output.num_player_2_gaps_filled}"
        )
        return postprocess_output

    def evaluate_results(self, player_bboxes: Dict[int, List[Optional[BoundingBox]]]):
        """Evaluate tracking results against ground truth.

        Args:
            player_bboxes: Dict mapping player_id to list of BoundingBox (one per frame)
        """
        print("Evaluating results...")

        num_frames = len(player_bboxes.get(1, []))

        for frame_idx in range(num_frames):
            frame_name = self.frame_name_formatter(frame_idx)

            # Extract bboxes for this frame as dict of [x1, y1, x2, y2]
            frame_bboxes = {}
            for player_id in [1, 2]:
                bbox_list = player_bboxes.get(player_id, [])
                if frame_idx < len(bbox_list) and bbox_list[frame_idx] is not None:
                    bbox = bbox_list[frame_idx]
                    frame_bboxes[player_id] = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]

            self.evaluate_frame(frame_bboxes, frame_name)

        print("Evaluation complete")
        return self.calculate_final_metrics()

    def visualize_results(
        self,
        df: pd.DataFrame,
        player_bboxes: Dict[int, List[Optional[BoundingBox]]],
        output_path: Optional[str] = None,
    ):
        """Create visualization video from results.

        Args:
            df: DataFrame with tracking data (positions)
            player_bboxes: Dict mapping player_id to list of bboxes
            output_path: Path to save the output video
        """
        if not output_path:
            return

        print(f"Creating visualization video: {output_path}")

        # Get video properties from first frame
        cap = cv2.VideoCapture(self.test_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read video for visualization")
            return

        # Setup video writer
        height, width = first_frame.shape[:2]
        codec = self.config["output"]["video_codec"]
        fourcc = cv2.VideoWriter_fourcc(*codec)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_idx, frame in enumerate(self.frame_generator()):
            frame_vis = frame.copy()

            # Draw tracking visualization
            for player_id in [1, 2]:
                # Get position from DataFrame
                if frame_idx in df.index:
                    pos_x = df.loc[frame_idx, f"player_{player_id}_x_pixel"]
                    pos_y = df.loc[frame_idx, f"player_{player_id}_y_pixel"]
                else:
                    pos_x = None
                    pos_y = None

                if pos_x is not None and pos_y is not None and not np.isnan(pos_x):
                    color_key = f"player_{player_id}_color"
                    color = tuple(self.config["visualization"][color_key])
                    radius = self.config["visualization"]["circle_radius"]
                    cv2.circle(frame_vis, (int(pos_x), int(pos_y)), radius, color, -1)

                    # Get bbox
                    bbox = None
                    bbox_list = player_bboxes.get(player_id, [])
                    if frame_idx < len(bbox_list):
                        bbox = bbox_list[frame_idx]

                    if bbox is not None:
                        thickness = self.config["visualization"]["bbox_thickness"]
                        cv2.rectangle(
                            frame_vis,
                            (int(bbox.x1), int(bbox.y1)),
                            (int(bbox.x2), int(bbox.y2)),
                            color,
                            thickness,
                        )
                        cv2.putText(
                            frame_vis,
                            f"P{player_id}",
                            (int(bbox.x1), int(bbox.y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

            video_writer.write(frame_vis)

        video_writer.release()
        print(f"Visualization video saved: {output_path}")

    def run_video(self):
        """Run complete evaluation pipeline with and without postprocessing"""
        # Process frames once
        print("\n" + "=" * 60)
        print("PROCESSING FRAMES")
        print("=" * 60)

        self.tracker.reset()
        df_raw, player_bboxes_raw = self.process_frames()

        # === EVALUATION WITHOUT POSTPROCESSING ===
        print("\n" + "=" * 60)
        print("EVALUATION WITHOUT POSTPROCESSING")
        print("=" * 60)

        self.reset_metrics()
        metrics_raw = self.evaluate_results(player_bboxes_raw)

        # Save raw results
        if self.output_results_path:
            raw_results_path = self.output_results_path.replace(".txt", "_raw.txt")
            self.save_results_to_txt(metrics_raw, raw_results_path)

        # Save raw visualization
        if self.output_video_path:
            raw_video_path = self.output_video_path.replace(".mp4", "_raw.mp4")
            self.visualize_results(
                df_raw, player_bboxes_raw, output_path=raw_video_path
            )

        # === APPLY POSTPROCESSING ===
        print("\n" + "=" * 60)
        print("APPLYING POSTPROCESSING")
        print("=" * 60)

        postprocess_output = self.apply_postprocessing(df_raw, player_bboxes_raw)

        # === EVALUATION WITH POSTPROCESSING ===
        # Evaluates using interpolated bounding boxes from postprocessing
        print("\n" + "=" * 60)
        print("EVALUATION WITH POSTPROCESSING")
        print("=" * 60)

        self.reset_metrics()
        metrics_postprocessed = self.evaluate_results(postprocess_output.player_bboxes)

        # Save postprocessed results
        if self.output_results_path:
            postprocessed_results_path = self.output_results_path.replace(
                ".txt", "_postprocessed.txt"
            )
            self.save_results_to_txt(metrics_postprocessed, postprocessed_results_path)

        # Save postprocessed visualization
        if self.output_video_path:
            postprocessed_video_path = self.output_video_path.replace(
                ".mp4", "_postprocessed.mp4"
            )
            self.visualize_results(
                postprocess_output.df,
                postprocess_output.player_bboxes,
                output_path=postprocessed_video_path,
            )

        return {"raw": metrics_raw, "postprocessed": metrics_postprocessed}

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
        player_1_id = self.config["evaluation"]["player_1_class_id"]
        player_2_id = self.config["evaluation"]["player_2_class_id"]

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

    def save_results_to_txt(self, results, output_path=None):
        """Save evaluation results to a TXT file"""
        if output_path is None:
            output_path = self.output_results_path

        if not output_path:
            return

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            overall = results["overall"]

            f.write("=" * 60 + "\n")
            f.write("TRACKING EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")

            f.write("OVERALL PERFORMANCE:\n")
            precision = self.config["output"]["results_precision"]
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

        print(f"Results saved to: {output_path}")

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

        # Print both raw and postprocessed results
        print("\n" + "=" * 60)
        print("SUMMARY - WITHOUT POSTPROCESSING")
        print("=" * 60)
        self.print_results(results["raw"])

        print("\n" + "=" * 60)
        print("SUMMARY - WITH POSTPROCESSING")
        print("=" * 60)
        self.print_results(results["postprocessed"])

        return results


if __name__ == "__main__":
    evaluator = PlayerTrackerEvaluator()
    evaluator.run_evaluation()
