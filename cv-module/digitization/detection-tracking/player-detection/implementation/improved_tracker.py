import cv2
import numpy as np
import torch
import torchvision
import time
from collections import deque
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights, detection
import json
import csv
from filterpy.kalman import KalmanFilter
import torch.backends.cudnn as cudnn
import numbers
import matplotlib.pyplot as plt

class CourtCalibrator:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.homography = None
        
    def _get_real_coords(self):
        return np.array(
            [[1.6, 5.44], [4.8, 5.44], [4.8, 7.04], [1.6, 7.04]],
            dtype=np.float32,
        )
        
    def detect_keypoints(self, frame):
        results = self.model(frame)[0]
        keypoints_dict = {}
        if results.keypoints is None or not hasattr(results.keypoints, "data"):
            return None
            
        keypoints_array = results.keypoints.data.cpu().numpy()
        for person_keypoints in keypoints_array:
            for idx, (x, y, conf) in enumerate(person_keypoints):
                if conf > 0.5:
                    keypoints_dict[idx] = (x, y)
                    
        if len(keypoints_dict) != 4:
            return None
            
        pixel_coords = np.array([keypoints_dict[i] for i in range(4)])
        return pixel_coords
        
    def compute_homography(self, frame):
        pixel_coords = self.detect_keypoints(frame)
        if pixel_coords is None:
            return None
            
        real_coords = self._get_real_coords()
        H, _ = cv2.findHomography(pixel_coords, real_coords)
        self.homography = H
        return H, pixel_coords
        
    def display_keypoints(self, frame, pixel_coords):
        """
        Display the frame with detected keypoints and optionally save the visualization.
        Args:
            frame: Input frame
            pixel_coords: Detected keypoints
        Returns:
            Visualization image with keypoints
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        if pixel_coords is None:
            cv2.putText(
                vis_frame,
                "Could not detect all 4 court keypoints",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            # Draw keypoints
            for x, y in pixel_coords:
                color = (0, 255, 0)
                # Convert to integer coordinates
                x, y = int(x), int(y)
                # Draw circle at keypoint
                cv2.circle(vis_frame, (x, y), 5, color, -1)
                
        return vis_frame
        
    def apply_homography(self, pt, homography):
        transformed_point = cv2.perspectiveTransform(
            np.array([[pt]], dtype=np.float32), homography
        )
        return transformed_point[0][0]
        
    def get_court_corners(self, pixel_coords=None):
        """Get the court corners in pixel coordinates"""
        if pixel_coords is not None:
            return pixel_coords
        elif hasattr(self, 'pixel_coords') and self.pixel_coords is not None:
            return self.pixel_coords
        return None


class SquashPlayerTracker:
    def __init__(self, video_path, output_path=None, max_history=30, reid_threshold=0.6, court_calibrator=None):
        self.video_path = video_path
        self.output_path = output_path
        self.max_history = max_history
        self.reid_threshold = reid_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.court_calibrator = court_calibrator

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Court information
        self.court_pixel_coords = None
        self.court_homography = None

        # Output files
        self.output_tracking_file = output_path.replace('.mp4', '_mot.txt') if output_path else None
        self.output_json_pose_file = output_path.replace('.mp4', '_poses.json') if output_path else None

        self.tracking_data = []
        self.pose_data = []

        # Enable cuDNN benchmark for performance
        cudnn.benchmark = True

        self.initialize_models()

        self.player_positions = {1: deque(maxlen=max_history), 2: deque(maxlen=max_history)}
        self.player_poses = {1: None, 2: None}
        self.player_features = {1: None, 2: None}
        self.player_colors = {1: (0, 255, 0), 2: (0, 0, 255)}

        # Initialize Kalman filters for smoothing player positions
        self.kalman_filters = {
            1: self._init_kalman_filter(),
            2: self._init_kalman_filter()
        }

        self.metrics = {
            'tracking_fps': [],
            'player_confidence': {1: [], 2: []},
            'detection_count': {1: 0, 2: 0},
            'reid_switches': 0,
            'lost_tracks': 0,
            'court_violations': {1: 0, 2: 0}  # New metric for tracking court violations
        }

        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
            )
        else:
            self.video_writer = None

    def initialize_models(self):
        print("Loading models...")
        # YOLO detector
        self.detector = YOLO('yolov8m.pt')

        # Using dummy pose model as a placeholder for Lightweight OpenPose
        # You can replace estimate_pose() with actual inference from a real lightweight pose model
        self.pose_model = None

        # Re-ID model (ResNet50 with 512 output features)
        self.reid_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.reid_model.fc = torch.nn.Linear(2048, 512)
        self.reid_model.eval()

        print("Models loaded successfully")

    def _init_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1000.
        kf.R = np.eye(2) * 10
        kf.Q = np.eye(4)
        return kf

    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None

        person_img = cv2.resize(person_img, (224, 224))
        person_tensor = torch.from_numpy(person_img).permute(2, 0, 1).float() / 255.0
        person_tensor = person_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.reid_model(person_tensor)
        return features.squeeze().cpu().numpy()

    def feature_distance(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return float('inf')
        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)
        return 1 - np.dot(feat1, feat2)

    def apply_court_constraints(self, detections):
        """
        Filter detections based on court boundaries and adjust tracking confidence.
        
        Args:
            detections: List of detections [x1, y1, x2, y2, conf, class]
            
        Returns:
            filtered_detections: Detections validated against court constraints
        """
        if self.court_pixel_coords is None:
            return detections
            
        # Extract court boundaries
        min_x = np.min(self.court_pixel_coords[:, 0])
        max_x = np.max(self.court_pixel_coords[:, 0])
        min_y = np.min(self.court_pixel_coords[:, 1])
        max_y = np.max(self.court_pixel_coords[:, 1])
        
        # Add some margin (players can be partially outside court)
        # margin_x = (max_x - min_x) * 0.15  # Increased margin for squash players who might lean outside court
        # margin_y = (max_y - min_y) * 0.1
        
        margin_x = (max_x - min_x)
        margin_y = (max_y - min_y) 

        filtered_detections = []
        for det in detections:
            bbox = det[0:4]
            # Get feet position (bottom center of bounding box)
            feet_x = (bbox[0] + bbox[2]) / 2
            feet_y = bbox[3]  # Bottom of bounding box
            
            # Check if feet are on/near the court
            in_court = (
                min_x - margin_x <= feet_x <= max_x + margin_x and
                min_y - margin_y <= feet_y <= max_y + margin_y
            )
            
            # Update confidence based on court position
            if in_court:
                # Slightly boost confidence for detections clearly in court
                conf_boost = 1.05 if (
                    min_x + margin_x/2 <= feet_x <= max_x - margin_x/2 and
                    min_y + margin_y/2 <= feet_y <= max_y - margin_y/2
                ) else 1.0
                
                det_copy = det.copy()
                det_copy[4] = min(det_copy[4] * conf_boost, 1.0)  # Cap at 1.0
                filtered_detections.append(det_copy)
            else:
                # If feet are far outside court, reduce confidence or skip
                distance_to_court = min(
                    abs(feet_x - min_x), abs(feet_x - max_x),
                    abs(feet_y - min_y), abs(feet_y - max_y)
                )
                # Only keep if detection is strong and not too far from court
                if det[4] > 0.6 and distance_to_court < max(margin_x, margin_y) * 3:
                    # Reduce confidence proportional to distance
                    det_copy = det.copy()
                    det_copy[4] *= max(0.5, 1.0 - (distance_to_court / (max(margin_x, margin_y) * 6)))
                    filtered_detections.append(det_copy)
        
        return filtered_detections

    def assign_detections_to_players(self, detections, frame):
        if len(detections) == 0:
            return {}
        assignments = {}
        det_features = [self.extract_features(frame, det[0:4]) for det in detections[:2]]

        if self.player_features[1] is None:
            for i in range(min(2, len(detections))):
                assignments[i] = i + 1
                self.player_features[i + 1] = det_features[i]
            return assignments

        matching_scores = {}
        for i, det in enumerate(detections[:2]):
            bbox = det[0:4]
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            for player_id in [1, 2]:
                reid_score = self.feature_distance(det_features[i], self.player_features[player_id])
                pos_score = 0
                if len(self.player_positions[player_id]) > 0:
                    last_pos = self.player_positions[player_id][-1]
                    pos_score = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2) / self.frame_width
                score = reid_score * 0.7 + pos_score * 0.3
                matching_scores[(i, player_id)] = score

        assigned_players = set()
        for (det_idx, player_id), score in sorted(matching_scores.items(), key=lambda x: x[1]):
            if det_idx in assignments or player_id in assigned_players:
                continue
            if score < self.reid_threshold:
                assignments[det_idx] = player_id
                assigned_players.add(player_id)
                if det_features[det_idx] is not None:
                    self.player_features[player_id] = (
                        0.7 * self.player_features[player_id] + 0.3 * det_features[det_idx]
                    )

        unassigned_dets = [i for i in range(min(2, len(detections))) if i not in assignments]
        unassigned_players = [i for i in [1, 2] if i not in assigned_players]
        if len(unassigned_dets) == 1 and len(unassigned_players) == 1:
            det_idx = unassigned_dets[0]
            player_id = unassigned_players[0]
            assignments[det_idx] = player_id
            self.player_features[player_id] = det_features[det_idx]
            self.metrics['reid_switches'] += 1

        return assignments

    def estimate_pose(self, frame, bbox):
        # Dummy lightweight pose estimation output with 17 COCO keypoints
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Each keypoint: [x, y, confidence], here all confidence=0.9
        return np.array([[cx, cy, 0.9]] * 17)

    def update_kalman_with_court_constraints(self, kf, measurement, player_id):
        """Apply physical constraints from court knowledge to Kalman filter"""
        if self.court_pixel_coords is None:
            kf.predict()
            kf.update(measurement)
            return kf
            
        # Extract court boundaries with margins
        min_x = np.min(self.court_pixel_coords[:, 0])
        max_x = np.max(self.court_pixel_coords[:, 0])
        min_y = np.min(self.court_pixel_coords[:, 1])
        max_y = np.max(self.court_pixel_coords[:, 1])
        
        # Add margin for player movement
        margin_x = (max_x - min_x) * 0.2
        margin_y = (max_y - min_y) * 0.15
        
        kf.predict()
        kf.update(measurement)
        
        # Check if the prediction is outside court boundaries
        if (kf.x[0] < min_x - margin_x or kf.x[0] > max_x + margin_x or 
            kf.x[1] < min_y - margin_y or kf.x[1] > max_y + margin_y):
            
            # Log potential court violation
            self.metrics['court_violations'][player_id] += 1
            
            # Clip the position to stay within expanded boundaries
            kf.x[0] = np.clip(kf.x[0], min_x - margin_x, max_x + margin_x)
            kf.x[1] = np.clip(kf.x[1], min_y - margin_y, max_y + margin_y)
            
            # Dampen velocity when near boundaries to prevent bouncing effects
            boundary_distance = min(
                abs(kf.x[0] - (min_x - margin_x)),
                abs(kf.x[0] - (max_x + margin_x)),
                abs(kf.x[1] - (min_y - margin_y)),
                abs(kf.x[1] - (max_y + margin_y))
            )
            
            if boundary_distance < max(margin_x, margin_y) * 0.1:
                # Apply stronger dampening when very close to boundaries
                velocity_scale = 0.7
                kf.x[2] *= velocity_scale  # x velocity
                kf.x[3] *= velocity_scale  # y velocity
        
        # Optional: constrain velocity based on typical squash movement speeds
        max_velocity = 20  # pixels per frame, adjust based on your video
        kf.x[2] = np.clip(kf.x[2], -max_velocity, max_velocity)  # x velocity
        kf.x[3] = np.clip(kf.x[3], -max_velocity, max_velocity)  # y velocity
        
        return kf

    def process_frame(self, frame):
        start_time = time.time()
        results = self.detector(frame)[0]
        detections = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        detections = np.hstack([detections, confidences[:, None], classes[:, None]])
        person_dets = [det for det in detections if det[5] == 0]  # class 0 = person
        
        # Apply court constraints if court coordinates are available
        if self.court_pixel_coords is not None:
            person_dets = self.apply_court_constraints(person_dets)
        
        person_dets = sorted(person_dets, key=lambda x: x[4], reverse=True)[:2]

        assignments = self.assign_detections_to_players(person_dets, frame)

        frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_tracking_entries = []
        frame_pose_entry = {"frame": frame_idx, "players": {}}

        # Draw court if available
        if self.court_pixel_coords is not None:
            frame = self.draw_court(frame, self.court_pixel_coords)

        for det_idx, player_id in assignments.items():
            bbox = person_dets[det_idx][0:4]
            confidence = person_dets[det_idx][4]

            # Kalman filter smoothing with court constraints
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            kf = self.kalman_filters[player_id]
            kf = self.update_kalman_with_court_constraints(kf, [cx, cy], player_id)
            x_smooth, y_smooth = kf.x[0], kf.x[1]

            self.player_positions[player_id].append((x_smooth, y_smooth))
            self.metrics['player_confidence'][player_id].append(confidence)
            self.metrics['detection_count'][player_id] += 1

            # Pose estimation (dummy placeholder)
            pose = self.estimate_pose(frame, bbox)
            self.player_poses[player_id] = pose

            color = self.player_colors[player_id]
            # Draw bbox
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # Label
            cv2.putText(frame, f"Player {player_id}", (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw pose
            if pose is not None:
                self.draw_pose(frame, pose, color)

            # Add to MOT tracking output (frame, id, x, y, w, h, conf, -1, -1, -1)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            frame_tracking_entries.append([
                frame_idx, player_id, int(x1), int(y1), int(w), int(h), round(float(confidence), 2), -1, -1, -1
            ])

            # Add to pose JSON output
            pose_entry = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(float(confidence), 2),
                "keypoints": pose.tolist() if pose is not None else []
            }
            frame_pose_entry["players"][str(player_id)] = pose_entry

            # Optional: Add real-world coordinates if homography is available
            if self.court_homography is not None:
                try:
                    # Get feet position in real-world coordinates
                    feet_pixel = (x_smooth, y_smooth + h/2)  # Bottom center of bounding box
                    real_coords = self.court_calibrator.apply_homography(feet_pixel, self.court_homography)
                    
                    # Add to pose data
                    pose_entry["real_world_position"] = [float(real_coords[0]), float(real_coords[1])]
                    
                    # Display real-world position
                    position_text = f"({real_coords[0]:.1f}, {real_coords[1]:.1f})"
                    cv2.putText(frame, position_text, (int(x_smooth), int(y_smooth) + int(h/2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    pass  # Handle potential errors in homography transformation

        self.tracking_data.extend(frame_tracking_entries)
        self.pose_data.append(frame_pose_entry)

        fps = 1.0 / (time.time() - start_time)
        self.metrics['tracking_fps'].append(fps)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return frame

    def draw_pose(self, frame, keypoints, color):
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    def draw_court(self, frame, court_coords):
        """Draw court boundaries on the frame"""
        if court_coords is None or len(court_coords) < 3:
            return frame
        
        # Draw court outline
        for i in range(len(court_coords)):
            cv2.line(frame, 
                     (int(court_coords[i][0]), int(court_coords[i][1])),
                     (int(court_coords[(i+1) % len(court_coords)][0]), int(court_coords[(i+1) % len(court_coords)][1])),
                     (255, 215, 0), 2)  # Gold color for court
        
        return frame

    def save_analysis_plot(self):
        plt.figure(figsize=(14, 8))

        # Prepare data
        frames = list(range(len(self.metrics['tracking_fps'])))
        
        # Detection confidence over time per player
        plt.subplot(2, 2, 1)
        for pid in [1, 2]:
            confs = self.metrics['player_confidence'][pid]
            plt.plot(range(len(confs)), confs, label=f'Player {pid}')
        plt.xlabel('Detections over time')
        plt.ylabel('Detection Confidence')
        plt.title('Detection Confidence Over Time')
        plt.legend()
        plt.grid(True)

        # Player trajectories (X vs Y smoothed)
        plt.subplot(2, 2, 2)
        for pid in [1, 2]:
            positions = np.array(self.player_positions[pid])
            if len(positions) > 0:
                plt.plot(positions[:, 0], positions[:, 1], label=f'Player {pid}')
                
        # Plot court boundaries if available
        if self.court_pixel_coords is not None:
            court_x = np.append(self.court_pixel_coords[:, 0], self.court_pixel_coords[0, 0])
            court_y = np.append(self.court_pixel_coords[:, 1], self.court_pixel_coords[0, 1])
            plt.plot(court_x, court_y, 'k--', label='Court')
            
        plt.gca().invert_yaxis()  # Invert y axis because image coords
        plt.xlabel('X position (smoothed)')
        plt.ylabel('Y position (smoothed)')
        plt.title('Player Trajectories')
        plt.legend()
        plt.grid(True)

        # Detection rate and court violations per player
        plt.subplot(2, 2, 3)
        total_frames = len(self.metrics['tracking_fps'])
        detection_rates = [self.metrics['detection_count'][pid] / total_frames if total_frames > 0 else 0 for pid in [1, 2]]
        
        # Create grouping for multi-metric bar chart
        x = np.arange(2)
        width = 0.35
        
        plt.bar(x - width/2, detection_rates, width, label='Detection Rate', color=['green', 'blue'])
        
        # Court violations normalized by detection count
        if total_frames > 0:
            violation_rates = [
                self.metrics['court_violations'][pid] / max(1, self.metrics['detection_count'][pid]) 
                for pid in [1, 2]
            ]
            plt.bar(x + width/2, violation_rates, width, label='Court Violations', color=['lightgreen', 'lightblue'])
        
        plt.xticks(x, ['Player 1', 'Player 2'])
        plt.ylabel('Rate')
        plt.title('Detection and Court Violation Rates')
        plt.legend()

        # Average confidence per player (bar plot)
        plt.subplot(2, 2, 4)
        avg_confs = [
            np.mean(self.metrics['player_confidence'][pid]) if len(self.metrics['player_confidence'][pid]) > 0 else 0 
            for pid in [1, 2]
        ]
        plt.bar(['Player 1', 'Player 2'], avg_confs, color=['green', 'blue'])
        plt.ylabel('Average Confidence')
        plt.title('Average Detection Confidence')

        plt.tight_layout()

        # Save plot to PNG
        if self.output_path:
            plot_path = self.output_path.replace('.mp4', '_analysis.png')
            plt.savefig(plot_path)
            print(f"Saved analysis plot to {plot_path}")
        plt.close()

    def run(self):
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return

        # First detect the court if we have a calibrator
        if self.court_calibrator:
            print("Detecting court...")
            ret, first_frame = self.cap.read()
            if ret:
                homography_result = self.court_calibrator.compute_homography(first_frame)
                if homography_result is not None:
                    self.court_homography, self.court_pixel_coords = homography_result
                    print("Court detected successfully!")
                    # Display the court detection
                    court_vis = self.court_calibrator.display_keypoints(first_frame, self.court_pixel_coords)
                    cv2.imshow("Court Detection", court_vis)
                    cv2.waitKey(1000)  # Show for 1 second
                else:
                    print("Warning: Failed to detect court. Continuing without court constraints.")
                
                # Reset video to start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Process and display frame
            processed_frame = self.process_frame(frame)
            cv2.imshow("Squash Player Tracking", processed_frame)

            if self.video_writer is not None:
                self.video_writer.write(processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and self.court_calibrator:
                # Recalibrate court on demand
                homography_result = self.court_calibrator.compute_homography(frame)
                if homography_result is not None:
                    self.court_homography, self.court_pixel_coords = homography_result
                    print("Court recalibrated!")

        self.cap.release()
        cv2.destroyAllWindows()
        self.save_outputs()
        self.save_analysis_plot()

        if self.video_writer is not None:
            self.video_writer.release()

    def save_outputs(self):
            if self.output_tracking_file:
                with open(self.output_tracking_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.tracking_data)
                print(f"Saved tracking output to {self.output_tracking_file}")

            if self.output_json_pose_file:
                with open(self.output_json_pose_file, 'w') as f:
                    json.dump(self.pose_data, f, indent=2)
                print(f"Saved pose output to {self.output_json_pose_file}")

            if self.output_path:
                # Save tracking metrics summary
                metrics_path = self.output_path.replace('.mp4', '_metrics.json')
                metrics_summary = {
                    'average_fps': np.mean(self.metrics['tracking_fps']),
                    'avg_confidence': {
                        '1': np.mean(self.metrics['player_confidence'][1]) if self.metrics['player_confidence'][1] else 0,
                        '2': np.mean(self.metrics['player_confidence'][2]) if self.metrics['player_confidence'][2] else 0
                    },
                    'detection_count': self.metrics['detection_count'],
                    'reid_switches': self.metrics['reid_switches'],
                    'lost_tracks': self.metrics['lost_tracks'],
                    'court_violations': self.metrics['court_violations'],
                    'total_frames': len(self.metrics['tracking_fps'])
                }
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_summary, f, indent=2)
                print(f"Saved metrics summary to {metrics_path}")
                
            print("All outputs saved successfully")

# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Squash Player Tracking')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default=None, help='Path to output video')
    parser.add_argument('--court_model', type=str, default='court_keypoints.pt', 
                        help='Path to YOLO court keypoint detection model')
    parser.add_argument('--history', type=int, default=30, help='Maximum history length for tracking')
    parser.add_argument('--reid_threshold', type=float, default=0.6, 
                        help='Threshold for re-identification')
    
    args = parser.parse_args()
    
    # Initialize court calibrator
    court_calibrator = CourtCalibrator(args.court_model)
    
    # Initialize and run tracker
    tracker = SquashPlayerTracker(
        video_path=args.video,
        output_path=args.output,
        max_history=args.history,
        reid_threshold=args.reid_threshold,
        court_calibrator=court_calibrator
    )
    
    tracker.run()