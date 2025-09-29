import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class SquashTrackerEvaluator:
    def __init__(self, coco_json_path, iou_threshold=0.5):
        self.coco_json_path = coco_json_path
        self.iou_threshold = iou_threshold
        self.ground_truth = self.load_coco_annotations()
        
        # Tracking evaluation metrics
        self.frame_results = []
        self.id_mapping = None  # Will store tracker_id -> ground_truth_class mapping
        self.mapping_confidence = 0.0
        
        # Accumulated metrics
        self.total_gt_boxes = 0
        self.total_pred_boxes = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.id_switches = 0
        self.track_breaks = 0
        
        # Per-player metrics - using category IDs 1 and 2 from COCO
        self.player_metrics = {1: {'tp': 0, 'fp': 0, 'fn': 0}, 
                              2: {'tp': 0, 'fp': 0, 'fn': 0}}
    
    def load_coco_annotations(self):
        """Load and organize COCO annotations by frame"""
        with open(self.coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"Loading COCO annotations...")
        print(f"Categories: {coco_data['categories']}")
        
        # Create mapping from image_id to frame data
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image_id
        annotations_by_frame = defaultdict(list)
        for ann in coco_data['annotations']:
            annotations_by_frame[ann['image_id']].append(ann)
        
        # Convert to frame-indexed structure
        ground_truth = {}
        for img_id, anns in annotations_by_frame.items():
            frame_name = images[img_id]['file_name']
            ground_truth[frame_name] = []
            
            for ann in anns:
                # Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2]
                x, y, w, h = ann['bbox']
                bbox = [x, y, x + w, y + h]
                ground_truth[frame_name].append({
                    'bbox': bbox,
                    'class_id': ann['category_id'],  # Keep original category_id (1 or 2)
                    'annotation_id': ann['id']
                })
        
        print(f"Loaded {len(ground_truth)} frames with annotations")
        return ground_truth
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection coordinates
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # Intersection area
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def establish_id_mapping(self, predictions, ground_truth_frame, frame_name):
        """Establish or verify the mapping between tracker IDs and ground truth classes"""
        if len(predictions) == 0 or len(ground_truth_frame) == 0:
            return
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truth_frame)))
        for i, pred in enumerate(predictions):
            pred_bbox = pred['bbox']
            for j, gt in enumerate(ground_truth_frame):
                gt_bbox = gt['bbox']
                iou_matrix[i, j] = self.calculate_iou(pred_bbox, gt_bbox)
        
        # Find best matches using Hungarian algorithm
        if iou_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            
            # Create potential mapping
            potential_mapping = {}
            mapping_ious = []
            
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] > self.iou_threshold:
                    tracker_id = predictions[row]['tracker_id']
                    gt_class = ground_truth_frame[col]['class_id']
                    potential_mapping[tracker_id] = gt_class
                    mapping_ious.append(iou_matrix[row, col])
            
            # If this is the first frame or mapping is more confident, update it
            current_confidence = np.mean(mapping_ious) if mapping_ious else 0
            if self.id_mapping is None or current_confidence > self.mapping_confidence:
                self.id_mapping = potential_mapping.copy()
                self.mapping_confidence = current_confidence
                print(f"Updated ID mapping at {frame_name}: {self.id_mapping} (confidence: {current_confidence:.3f})")
    
    def evaluate_frame(self, predictions, frame_name):
        """Evaluate a single frame against ground truth"""
        if frame_name not in self.ground_truth:
            print(f"Warning: No ground truth for frame {frame_name}")
            return
        
        gt_frame = self.ground_truth[frame_name]
        self.total_gt_boxes += len(gt_frame)
        self.total_pred_boxes += len(predictions)
        
        # Establish/verify ID mapping if needed
        self.establish_id_mapping(predictions, gt_frame, frame_name)
        
        if self.id_mapping is None:
            print("Warning: Could not establish ID mapping")
            return
        
        # Convert predictions to use ground truth class IDs
        mapped_predictions = []
        for pred in predictions:
            tracker_id = pred['tracker_id']
            if tracker_id in self.id_mapping:
                mapped_pred = pred.copy()
                mapped_pred['class_id'] = self.id_mapping[tracker_id]
                mapped_predictions.append(mapped_pred)
        
        # Calculate IoU matrix between mapped predictions and ground truth
        iou_matrix = np.zeros((len(mapped_predictions), len(gt_frame)))
        for i, pred in enumerate(mapped_predictions):
            for j, gt in enumerate(gt_frame):
                iou_matrix[i, j] = self.calculate_iou(pred['bbox'], gt['bbox'])
        
        # Match predictions to ground truth
        matched_pred = set()
        matched_gt = set()
        
        if iou_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] > self.iou_threshold:
                    pred = mapped_predictions[row]
                    gt = gt_frame[col]
                    
                    # Check if class IDs match
                    if pred['class_id'] == gt['class_id']:
                        self.true_positives += 1
                        self.player_metrics[gt['class_id']]['tp'] += 1
                    else:
                        # Wrong ID assignment
                        self.false_positives += 1
                        self.false_negatives += 1
                        self.id_switches += 1
                        self.player_metrics[pred['class_id']]['fp'] += 1
                        self.player_metrics[gt['class_id']]['fn'] += 1
                    
                    matched_pred.add(row)
                    matched_gt.add(col)
        
        # Unmatched predictions are false positives
        for i in range(len(mapped_predictions)):
            if i not in matched_pred:
                self.false_positives += 1
                pred_class = mapped_predictions[i]['class_id']
                if pred_class in self.player_metrics:
                    self.player_metrics[pred_class]['fp'] += 1
        
        # Unmatched ground truth are false negatives
        for j in range(len(gt_frame)):
            if j not in matched_gt:
                self.false_negatives += 1
                gt_class = gt_frame[j]['class_id']
                if gt_class in self.player_metrics:
                    self.player_metrics[gt_class]['fn'] += 1
        
        # Store frame result for detailed analysis
        self.frame_results.append({
            'frame_name': frame_name,
            'predictions': len(predictions),
            'ground_truth': len(gt_frame),
            'true_positives': len(matched_pred) if 'matched_pred' in locals() else 0,
            'id_mapping_used': self.id_mapping.copy() if self.id_mapping else None
        })
    
    def calculate_final_metrics(self):
        """Calculate final tracking metrics"""
        if self.total_gt_boxes == 0:
            return {}
        
        # Overall metrics
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MOTA (Multiple Object Tracking Accuracy)
        mota = 1 - (self.false_negatives + self.false_positives + self.id_switches) / self.total_gt_boxes if self.total_gt_boxes > 0 else 0
        
        # Per-player metrics - using category IDs 1 and 2
        player_stats = {}
        for player_id in [1, 2]:
            if player_id in self.player_metrics:
                metrics = self.player_metrics[player_id]
                tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
                
                p_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                p_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                p_f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall) if (p_precision + p_recall) > 0 else 0
                
                player_stats[f'player_{player_id}'] = {
                    'precision': p_precision,
                    'recall': p_recall,
                    'f1_score': p_f1,
                    'detections': tp + fn,  # Total ground truth instances
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn
                }
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'mota': mota,
                'total_frames': len(self.frame_results),
                'id_switches': self.id_switches,
                'id_mapping': self.id_mapping,
                'mapping_confidence': self.mapping_confidence
            },
            'per_player': player_stats,
            'detection_stats': {
                'total_ground_truth': self.total_gt_boxes,
                'total_predictions': self.total_pred_boxes,
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives
            }
        }