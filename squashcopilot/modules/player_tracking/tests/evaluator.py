"""
Player Tracker Evaluator

Evaluates player tracking performance with score diagnostics.
Optionally evaluates against COCO-format ground truth annotations.
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
    def __init__(self, config: dict = None, evaluate_ground_truth: bool = False, max_frames: int = None):
        """Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            evaluate_ground_truth: If True, load ground truth and run accuracy evaluation
            max_frames: Maximum number of frames to process (overrides config). None = process all frames
        """
        if config is None:
            # Load the tests section from the player_tracking config
            full_config = load_config(config_name="player_tracking")
            config = full_config["tests"]
        self.config = config
        self.evaluate_ground_truth = evaluate_ground_truth
        
        # Override max_frames if provided
        if max_frames is not None:
            self.config["processing"]["max_frames"] = max_frames

        # Get test directory path
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

        # Resolve paths relative to test directory
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

        # Initialize tracker and calibrator
        self.tracker = PlayerTracker()
        self.calibrator = CourtCalibrator()
        self.calibration = None

        # Ground truth data (only loaded if evaluate_ground_truth is True)
        self.ground_truth = {}
        self.frame_name_formatter = None
        
        if self.evaluate_ground_truth:
            self.coco_json_path = os.path.join(
                self.test_dir, self.config["paths"]["coco_annotations"]
            )
            self._load_ground_truth()
            self._initialize_metrics()
    
    def _load_ground_truth(self):
        """Load COCO-format ground truth annotations."""
        print(f"Loading ground truth from: {self.coco_json_path}")
        with open(self.coco_json_path, "r") as f:
            coco_data = json.load(f)

        # Parse annotations
        for img_data in coco_data["images"]:
            frame_name = img_data["file_name"]
            self.ground_truth[frame_name] = []

        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            img_data = next(
                (img for img in coco_data["images"] if img["id"] == img_id), None
            )
            if img_data:
                frame_name = img_data["file_name"]
                x, y, w, h = ann["bbox"]
                self.ground_truth[frame_name].append(
                    {
                        "bbox": [x, y, x + w, y + h],
                        "class_id": ann["category_id"],
                    }
                )

        # Set up frame name formatter
        frame_pattern = self.config.get("frame_name_pattern", "frame_{:06d}.jpg")
        self.frame_name_formatter = lambda idx: frame_pattern.format(idx)

        print(f"Loaded ground truth for {len(self.ground_truth)} frames")

    def _initialize_metrics(self):
        """Initialize evaluation metrics."""
        self.iou_threshold = self.config["evaluation"]["iou_threshold"]
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_gt_boxes = 0
        self.total_pred_boxes = 0
        self.id_switches = 0
        self.id_mapping = None
        self.mapping_confidence = 0
        self.frame_results = []
        self.player_metrics = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "detections": 0}
        )

    def reset_metrics(self):
        """Reset all metrics to zero."""
        if self.evaluate_ground_truth:
            self._initialize_metrics()

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update_id_mapping(self, player_bboxes, ground_truth_frame, frame_name):
        """Update ID mapping between tracker IDs and ground truth classes."""
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
        """Evaluate a single frame against ground truth."""
        if frame_name not in self.ground_truth:
            return

        ground_truth_frame = self.ground_truth[frame_name]
        self.total_gt_boxes += len(ground_truth_frame)
        self.total_pred_boxes += len(player_bboxes)

        # Update ID mapping
        self.update_id_mapping(player_bboxes, ground_truth_frame, frame_name)

        if not self.id_mapping:
            return

        # Match predictions with ground truth using ID mapping
        matched_pred = set()
        matched_gt = set()

        for player_id, bbox in player_bboxes.items():
            if player_id not in self.id_mapping:
                continue

            gt_class = self.id_mapping[player_id]
            best_iou = 0
            best_gt_idx = None

            for j, gt in enumerate(ground_truth_frame):
                if gt["class_id"] == gt_class and j not in matched_gt:
                    iou = self.calculate_iou(bbox, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_gt_idx is not None and best_iou > self.iou_threshold:
                matched_pred.add(player_id)
                matched_gt.add(best_gt_idx)
                self.true_positives += 1
                self.player_metrics[gt_class]["tp"] += 1

        # Count false positives and false negatives
        for player_id in player_bboxes.keys():
            if player_id not in matched_pred:
                self.false_positives += 1
                if player_id in self.id_mapping:
                    gt_class = self.id_mapping[player_id]
                    if gt_class in self.player_metrics:
                        self.player_metrics[gt_class]["fp"] += 1

        for j, gt in enumerate(ground_truth_frame):
            if j not in matched_gt:
                self.false_negatives += 1
                gt_class = gt["class_id"]
                if gt_class in self.player_metrics:
                    self.player_metrics[gt_class]["fn"] += 1

        self.frame_results.append(
            {
                "frame_name": frame_name,
                "predictions": len(player_bboxes),
                "ground_truth": len(ground_truth_frame),
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

        # Enable score diagnostics
        self.tracker.enable_score_diagnostics()

        for frame_idx, frame_img in tqdm.tqdm(
            enumerate(self.frame_generator()),
            total=total_frames,
            desc="Processing frames",
        ):
            # Calibrate on first frame
            if frame_idx == 0:
                calibration_input = CourtCalibrationInput(
                    frame=Frame(image=frame_img, frame_number=0, timestamp=0.0)
                )
                self.calibration = self.calibrator.process_frame(calibration_input)

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

        # Get score statistics
        score_stats = self.tracker.get_score_statistics()
        self.print_score_statistics(score_stats)

        # Convert to DataFrame + complex data using the model function
        df, complex_data = player_tracking_outputs_to_dataframe(results_list)

        # Build player_bboxes dict from complex_data
        player_bboxes = {
            1: complex_data["player_1_bboxes"],
            2: complex_data["player_2_bboxes"],
        }

        return df, player_bboxes

    def print_score_statistics(self, stats):
        """Print reid vs position score statistics."""
        if stats is None:
            print("\nNo score statistics available (no tracking occurred)")
            return

        print("\n" + "=" * 70)
        print("REID vs POSITION SCORE DIAGNOSTICS")
        print("=" * 70)

        reid_weight = self.config.get("tracker", {}).get("reid_weight", 0.1)
        pos_weight = self.config.get("tracker", {}).get("position_weight", 0.9)

        print(f"\nCONFIGURED WEIGHTS:")
        print(f"   Reid weight:     {reid_weight:.2f}")
        print(f"   Position weight: {pos_weight:.2f}")

        print(f"\nREID SCORES (Cosine Distance, range 0-2):")
        reid = stats['reid']
        if reid['count'] > 0:
            print(f"   Count:  {reid['count']}")
            print(f"   Min:    {reid['min']:.6f}")
            print(f"   Max:    {reid['max']:.6f}")
            print(f"   Mean:   {reid['mean']:.6f}")
            print(f"   Median: {reid['median']:.6f}")
            print(f"   Std:    {reid['std']:.6f}")
        else:
            print("   No valid reid scores recorded")

        print(f"\nPOSITION SCORES (Normalized pixel distance / frame_width, range 0-1+):")
        pos = stats['position']
        if pos['count'] > 0:
            print(f"   Count:  {pos['count']}")
            print(f"   Min:    {pos['min']:.6f}")
            print(f"   Max:    {pos['max']:.6f}")
            print(f"   Mean:   {pos['mean']:.6f}")
            print(f"   Median: {pos['median']:.6f}")
            print(f"   Std:    {pos['std']:.6f}")
        else:
            print("   No valid position scores recorded")

        # Calculate weighted contributions
        if reid['count'] > 0 and pos['count'] > 0:
            print(f"\nWEIGHTED CONTRIBUTIONS TO FINAL SCORE:")
            reid_contribution = reid['mean'] * reid_weight
            pos_contribution = pos['mean'] * pos_weight
            total = reid_contribution + pos_contribution
            
            print(f"   Reid contribution:     {reid_contribution:.6f} ({reid_contribution/total*100:.1f}%)")
            print(f"   Position contribution: {pos_contribution:.6f} ({pos_contribution/total*100:.1f}%)")
            print(f"   Total mean score:      {total:.6f}")
            
            print(f"\nANALYSIS:")
            if reid_contribution > pos_contribution * 2:
                print("   ⚠️  Reid DOMINATES matching (>2x position contribution)")
                print("   → Position-based tracking is being ignored")
            elif pos_contribution > reid_contribution * 2:
                print("   ✓ Position DOMINATES matching (>2x reid contribution)")
                print("   → Reid features may not be helping")
            else:
                print("   ✓ Reid and position are balanced")

            # Check if position scores are suspiciously low
            if pos['mean'] < 0.01:
                print(f"\n   ⚠️  Position scores are VERY LOW (mean={pos['mean']:.6f})")
                print("   → Players barely move between frames")
                print("   → Position-based tracking cannot distinguish between players")

        print("=" * 70 + "\n")

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
        if not self.evaluate_ground_truth:
            return None

        print("Evaluating results against ground truth...")

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

    def calculate_final_metrics(self):
        """Calculate final evaluation metrics."""
        if not self.evaluate_ground_truth:
            return None

        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        mota = 1 - (fp + fn + self.id_switches) / self.total_gt_boxes

        player_stats = {}
        for player_id in [1, 2]:
            if player_id in self.player_metrics:
                metrics = self.player_metrics[player_id]
                tp = metrics["tp"]
                fp = metrics["fp"]
                fn = metrics["fn"]

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

    def visualize_results(
        self,
        df: pd.DataFrame,
        player_bboxes: Dict[int, List[Optional[BoundingBox]]],
        output_path: Optional[str] = None,
    ):
        """Create visualization video from results with reid/position scores overlay.

        Args:
            df: DataFrame with tracking data
            player_bboxes: Dict mapping player_id to list of bboxes
            output_path: Output video path
        """
        if output_path is None:
            output_path = self.output_video_path

        if not output_path:
            return

        print(f"Creating visualization video: {output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cap = cv2.VideoCapture(self.test_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        max_frames = self.config["processing"]["max_frames"]
        frame_count = 0

        # Get score logs from tracker
        reid_scores = self.tracker.score_logs.get('reid_scores', [])
        pos_scores = self.tracker.score_logs.get('pos_scores', [])
        
        # Get weights from config
        reid_weight = self.tracker.config["tracker"]["reid_weight"]
        pos_weight = self.tracker.config["tracker"]["position_weight"]
        
        # Track score index (4 scores per frame: 2 detections x 2 players)
        score_idx = 0

        while cap.isOpened():
            if max_frames and frame_count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_vis = frame.copy()

            # Draw bounding boxes
            colors = {1: (0, 255, 0), 2: (255, 0, 0)}  # Player 1: Green, Player 2: Red
            
            # Collect scores for this frame
            frame_reid_scores = []
            frame_pos_scores = []
            frame_combined_scores = []
            
            if score_idx < len(reid_scores) - 3:  # Make sure we have scores available
                # Get the 4 scores for this frame (2 detections x 2 players)
                frame_reid_scores = reid_scores[score_idx:score_idx+4]
                frame_pos_scores = pos_scores[score_idx:score_idx+4]
                frame_combined_scores = [
                    r * reid_weight + p * pos_weight 
                    for r, p in zip(frame_reid_scores, frame_pos_scores)
                ]
                score_idx += 4

            for player_id in [1, 2]:
                bbox_list = player_bboxes.get(player_id, [])
                if frame_count < len(bbox_list) and bbox_list[frame_count] is not None:
                    bbox = bbox_list[frame_count]
                    color = colors[player_id]
                    thickness = 2

                    # Draw rectangle
                    cv2.rectangle(
                        frame_vis,
                        (int(bbox.x1), int(bbox.y1)),
                        (int(bbox.x2), int(bbox.y2)),
                        color,
                        thickness,
                    )
                    
                    # Draw player ID
                    cv2.putText(
                        frame_vis,
                        f"P{player_id}",
                        (int(bbox.x1), int(bbox.y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # Draw score overlay panel
            if frame_reid_scores and frame_pos_scores:
                # Create semi-transparent overlay panel
                overlay = frame_vis.copy()
                panel_height = 180
                panel_width = 400
                panel_x = 10
                panel_y = 10
                
                cv2.rectangle(
                    overlay,
                    (panel_x, panel_y),
                    (panel_x + panel_width, panel_y + panel_height),
                    (0, 0, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.7, frame_vis, 0.3, 0, frame_vis)
                
                # Draw panel border
                cv2.rectangle(
                    frame_vis,
                    (panel_x, panel_y),
                    (panel_x + panel_width, panel_y + panel_height),
                    (255, 255, 255),
                    2
                )
                
                # Title
                cv2.putText(
                    frame_vis,
                    "Matching Scores",
                    (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                
                # Weights info
                cv2.putText(
                    frame_vis,
                    f"Weights: Reid={reid_weight:.2f} | Pos={pos_weight:.2f}",
                    (panel_x + 10, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                )
                
                # Show scores (assuming 2 detections matched to 2 players)
                y_offset = panel_y + 75
                line_height = 25
                
                # Average scores for simplicity (since we have 4 scores per frame)
                if len(frame_reid_scores) >= 4:
                    avg_reid = np.mean(frame_reid_scores)
                    avg_pos = np.mean(frame_pos_scores)
                    avg_combined = np.mean(frame_combined_scores)
                    
                    cv2.putText(
                        frame_vis,
                        f"Reid Score:     {avg_reid:.4f}",
                        (panel_x + 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (100, 200, 255),
                        1,
                    )
                    
                    cv2.putText(
                        frame_vis,
                        f"Position Score: {avg_pos:.4f}",
                        (panel_x + 15, y_offset + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (100, 255, 100),
                        1,
                    )
                    
                    cv2.putText(
                        frame_vis,
                        f"Combined Score: {avg_combined:.4f}",
                        (panel_x + 15, y_offset + line_height * 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )
                    
                    # Show weighted contributions
                    reid_contrib = avg_reid * reid_weight
                    pos_contrib = avg_pos * pos_weight
                    total = reid_contrib + pos_contrib
                    
                    if total > 0:
                        reid_pct = (reid_contrib / total) * 100
                        pos_pct = (pos_contrib / total) * 100
                        
                        cv2.putText(
                            frame_vis,
                            f"Reid: {reid_pct:.1f}% | Pos: {pos_pct:.1f}%",
                            (panel_x + 15, y_offset + line_height * 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )

            # Add frame counter
            cv2.putText(
                frame_vis,
                f"Frame: {frame_count}",
                (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            video_writer.write(frame_vis)
            frame_count += 1

        cap.release()
        video_writer.release()
        print(f"Visualization video saved: {output_path}")

    def save_results_to_txt(self, results, output_path=None):
        """Save evaluation results to a TXT file"""
        if not self.evaluate_ground_truth or results is None:
            return

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
        if not self.evaluate_ground_truth or results is None:
            return

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

    def run_video(self):
        """Run complete evaluation pipeline with and without postprocessing"""
        # Process frames once
        print("\n" + "=" * 60)
        print("PROCESSING FRAMES")
        print("=" * 60)

        self.tracker.reset()
        df_raw, player_bboxes_raw = self.process_frames()

        results = {}

        # === EVALUATION WITHOUT POSTPROCESSING ===
        if self.evaluate_ground_truth:
            print("\n" + "=" * 60)
            print("EVALUATION WITHOUT POSTPROCESSING")
            print("=" * 60)

            self.reset_metrics()
            metrics_raw = self.evaluate_results(player_bboxes_raw)
            results["raw"] = metrics_raw

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
        if self.evaluate_ground_truth:
            print("\n" + "=" * 60)
            print("EVALUATION WITH POSTPROCESSING")
            print("=" * 60)

            self.reset_metrics()
            metrics_postprocessed = self.evaluate_results(
                postprocess_output.player_bboxes
            )
            results["postprocessed"] = metrics_postprocessed

            # Save postprocessed results
            if self.output_results_path:
                postprocessed_results_path = self.output_results_path.replace(
                    ".txt", "_postprocessed.txt"
                )
                self.save_results_to_txt(
                    metrics_postprocessed, postprocessed_results_path
                )

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

        return results

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        results = self.run_video()

        if self.evaluate_ground_truth:
            # Print both raw and postprocessed results
            print("\n" + "=" * 60)
            print("SUMMARY - WITHOUT POSTPROCESSING")
            print("=" * 60)
            self.print_results(results.get("raw"))

            print("\n" + "=" * 60)
            print("SUMMARY - WITH POSTPROCESSING")
            print("=" * 60)
            self.print_results(results.get("postprocessed"))

        return results


if __name__ == "__main__":
    # Run with score diagnostics only (no ground truth evaluation)
    evaluator = PlayerTrackerEvaluator(evaluate_ground_truth=False, max_frames=1000)
    evaluator.run_evaluation()
    
    # To run with ground truth evaluation, use:
    # evaluator = PlayerTrackerEvaluator(evaluate_ground_truth=True)
    # evaluator.run_evaluation()