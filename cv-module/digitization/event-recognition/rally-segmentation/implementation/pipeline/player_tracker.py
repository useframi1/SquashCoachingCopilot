from ultralytics import YOLO
import numpy as np
import cv2
import torch
from collections import deque
from filterpy.kalman import KalmanFilter


class PlayerTracker:
    def __init__(
        self,
        homography,
        config,
        max_history=100,
        reid_threshold=0.6,
    ):
        # From original PlayerTracker
        self.config = config
        self.homography = homography
        self.player_real_positions = {}
        self.model = YOLO(self.config["model_path"])
        self.player_class_id = self.config["player_class_id"]

        # From SquashPlayerTracker
        self.max_history = max_history
        self.reid_threshold = reid_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Court information
        self.court_pixel_coords = None

        # Initialize player tracking data structures
        self.player_positions = {}  # Dictionary of player positions (pixel coords)
        self.player_features = {}  # For re-identification
        self.player_colors = {}  # For visualization
        self.kalman_filters = {}  # For motion smoothing

        self.initialize_homography(homography)

        # Metrics tracking
        self.metrics = {
            "tracking_fps": [],
            "player_confidence": {},
            "detection_count": {},
            "reid_switches": 0,
            "lost_tracks": 0,
            "court_violations": {},
        }

        # Initialize the feature extractor for re-ID if needed
        self.initialize_reid_model()

    def initialize_homography(self, homography):
        """Store homography and compute its inverse"""
        self.homography = homography

        # Compute inverse homography for converting back to pixel coordinates
        if homography is not None:
            try:
                self.inv_homography = np.linalg.inv(homography)
            except:
                self.inv_homography = None
                print("Warning: Could not compute inverse homography")

    def initialize_reid_model(self):
        """Initialize a lightweight model for player re-identification"""
        try:
            # Simple ResNet18 feature extractor or similar lightweight model
            from torchvision.models import resnet18, ResNet18_Weights

            # Load the model
            self.reid_model = resnet18(weights=ResNet18_Weights.DEFAULT)

            # Replace the final layer BEFORE moving to device
            self.reid_model.fc = torch.nn.Linear(512, 128)

            # Move the entire model to device
            self.reid_model = self.reid_model.to(self.device)
            self.reid_model.eval()

            self.use_reid = True

        except Exception as e:
            print(f"Warning: Could not initialize re-ID model: {e}")
            import traceback

            traceback.print_exc()
            self.use_reid = False

    def _init_kalman_filter(self):
        """Initialize a Kalman filter for tracking a player"""
        kf = KalmanFilter(
            dim_x=4, dim_z=2
        )  # State: [x, y, vx, vy], Measurement: [x, y]
        kf.F = np.array(
            [
                [1, 0, 1, 0],  # x = x + vx
                [0, 1, 0, 1],  # y = y + vy
                [0, 0, 1, 0],  # vx = vx
                [0, 0, 0, 1],  # vy = vy
            ]
        )
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observe x  # Observe y
        kf.P *= 1000.0  # Initial uncertainty
        kf.R = np.eye(2) * 10  # Measurement noise
        kf.Q = np.eye(4) * 0.1  # Process noise
        return kf

    def extract_features(self, frame, bbox):
        """Extract features for player re-identification"""
        if not self.use_reid:
            return None

        try:
            x1, y1, w, h = bbox  # XYWH format
            x2, y2 = x1 + w, y1 + h

            # Ensure valid crop area
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))

            if x2 <= x1 or y2 <= y1:
                return None  # Invalid bbox

            # Crop and preprocess
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None

            # Resize and normalize
            person_img = cv2.resize(person_img, (224, 224))

            # Convert BGR to RGB (OpenCV uses BGR, PyTorch models expect RGB)
            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # Convert to tensor and normalize to [0, 1]
            person_tensor = (
                torch.from_numpy(person_img).permute(2, 0, 1).float() / 255.0
            )

            # Apply ImageNet normalization
            normalize = (
                torch.nn.functional.normalize
                if hasattr(torch.nn.functional, "normalize")
                else lambda x: x
            )
            # Standard ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            # Normalize
            person_tensor = (person_tensor - mean) / std

            # Add batch dimension and move to the same device as the model
            person_tensor = person_tensor.unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.reid_model(person_tensor)

            return features.squeeze().cpu().numpy()

        except Exception as e:
            print(f"Feature extraction error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def feature_distance(self, feat1, feat2):
        """Calculate distance between feature vectors"""
        if feat1 is None or feat2 is None:
            return float("inf")

        # Normalize and compute cosine distance
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-6)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-6)
        return 1 - np.dot(feat1, feat2)

    def apply_homography(self, pt):
        """Transform pixel coordinates to real-world coordinates"""
        if self.homography is None:
            return None

        transformed_point = cv2.perspectiveTransform(
            np.array([[pt]], dtype=np.float32), self.homography
        )
        return transformed_point[0][0]

    def update_player_real_positions(self, pid, real_xy):
        """Add real-world coordinates to player history"""
        if pid not in self.player_real_positions:
            self.player_real_positions[pid] = []
        self.player_real_positions[pid].append(real_xy)

        # Keep the history limited
        if len(self.player_real_positions[pid]) > self.max_history:
            self.player_real_positions[pid].pop(0)

    def apply_court_constraints(self, detections):
        """
        Filter detections based on court boundaries in real-world coordinates

        Args:
            detections: List of detections [bbox, conf, class_name]

        Returns:
            filtered_detections: Detections that are valid based on court constraints
        """
        if self.homography is None:
            return detections

        # Define real-world court boundaries in meters
        # These values should match your real squash court dimensions
        # Standard squash court: 9.75m x 6.4m
        min_x_real = 0.0  # left boundary in meters
        max_x_real = 6.4  # right boundary in meters
        min_y_real = 0.0  # front (front wall) boundary in meters
        max_y_real = 9.75  # back boundary in meters

        # Add margin for player movement outside court
        margin_x_real = 1.0  # 1 meter margin on sides
        margin_y_real = 1.0  # 1 meter margin front/back

        filtered_detections = []

        for det in detections:
            bbox, conf, class_name = det
            x, y, w, h = bbox

            # Get feet position (bottom center of bounding box)
            feet_x_pixel = x + w / 2
            feet_y_pixel = y + h

            # Transform to real-world coordinates
            feet_pos_pixel = (feet_x_pixel, feet_y_pixel)
            try:
                # Apply homography to get real-world coordinates
                feet_pos_real = self.apply_homography(feet_pos_pixel)

                if feet_pos_real is None:
                    # If homography fails, keep the detection based on confidence
                    if conf > 0.6:
                        filtered_detections.append(det)
                    continue

                feet_x_real, feet_y_real = feet_pos_real

                # Check if feet are on/near the court in real-world coordinates
                in_court = (
                    min_x_real - margin_x_real
                    <= feet_x_real
                    <= max_x_real + margin_x_real
                    and min_y_real - margin_y_real
                    <= feet_y_real
                    <= max_y_real + margin_y_real
                )

                # Update confidence based on court position
                if in_court:
                    # Boost confidence for detections clearly in court
                    center_boost = (
                        1.05
                        if (
                            min_x_real + margin_x_real / 2
                            <= feet_x_real
                            <= max_x_real - margin_x_real / 2
                            and min_y_real + margin_y_real / 2
                            <= feet_y_real
                            <= max_y_real - margin_y_real / 2
                        )
                        else 1.0
                    )

                    det_copy = list(det)
                    det_copy[1] = min(det[1] * center_boost, 1.0)  # Cap at 1.0
                    filtered_detections.append(tuple(det_copy))
                else:
                    # If feet are outside court, calculate distance to court
                    distance_to_court = min(
                        abs(feet_x_real - min_x_real),
                        abs(feet_x_real - max_x_real),
                        abs(feet_y_real - min_y_real),
                        abs(feet_y_real - max_y_real),
                    )

                    # Only keep if detection is strong and not too far from court
                    if (
                        conf > 0.6
                        and distance_to_court < max(margin_x_real, margin_y_real) * 2
                    ):
                        # Reduce confidence proportional to distance
                        det_copy = list(det)
                        det_copy[1] *= max(
                            0.5,
                            1.0
                            - (
                                distance_to_court
                                / (max(margin_x_real, margin_y_real) * 4)
                            ),
                        )
                        filtered_detections.append(tuple(det_copy))

            except Exception as e:
                # If there's an error in homography, keep detection if confidence is high
                if conf > 0.7:
                    filtered_detections.append(det)

        return filtered_detections

    def assign_players(self, detections, frame):
        """Assign detections to player IDs using re-ID features and tracking"""
        player_assignments = {}

        if not self.use_reid or not detections:
            return player_assignments

        # Extract features for all detections
        det_features = []
        for det in detections:
            bbox, _, _ = det
            features = self.extract_features(frame, bbox)
            det_features.append(features)

        # If first frame, initialize players
        if not self.player_features:
            for i, det in enumerate(detections[:2]):  # Assign first two detections
                player_id = i + 1  # Start IDs from 1
                bbox, conf, _ = det
                player_assignments[i] = player_id
                self.player_features[player_id] = det_features[i]
                self.player_positions[player_id] = deque(maxlen=self.max_history)
                self.kalman_filters[player_id] = self._init_kalman_filter()
                self.player_colors[player_id] = (
                    (0, 255, 0) if player_id == 1 else (0, 0, 255)
                )
                self.metrics["player_confidence"][player_id] = []
                self.metrics["detection_count"][player_id] = 0
                self.metrics["court_violations"][player_id] = 0
            return player_assignments

        # Calculate matching scores based on appearance and position
        matching_scores = {}
        for i, det in enumerate(detections[: min(2, len(detections))]):
            bbox, conf, _ = det
            x, y, w, h = bbox
            center = (x + w / 2, y + h / 2)

            for player_id in self.player_features.keys():
                # Re-ID score
                reid_score = self.feature_distance(
                    det_features[i], self.player_features[player_id]
                )

                # Position score based on last known position
                pos_score = float("inf")
                if (
                    player_id in self.player_positions
                    and self.player_positions[player_id]
                ):
                    last_pos = self.player_positions[player_id][-1]
                    pos_score = np.sqrt(
                        (center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2
                    )
                    # Normalize by image width
                    pos_score = pos_score / 1000.0  # Rough normalization

                # Combined score (lower is better)
                score = reid_score * 0.7 + pos_score * 0.3
                matching_scores[(i, player_id)] = score

        # Assign detections to players based on scores
        assigned_players = set()
        for (det_idx, player_id), score in sorted(
            matching_scores.items(), key=lambda x: x[1]
        ):
            if det_idx in player_assignments or player_id in assigned_players:
                continue

            if score < self.reid_threshold:
                player_assignments[det_idx] = player_id
                assigned_players.add(player_id)

                # Update feature representation with moving average
                if (
                    det_features[det_idx] is not None
                    and self.player_features[player_id] is not None
                ):
                    self.player_features[player_id] = (
                        0.7 * self.player_features[player_id]
                        + 0.3 * det_features[det_idx]
                    )

        # Handle unassigned detections/players
        unassigned_dets = [
            i for i in range(min(2, len(detections))) if i not in player_assignments
        ]
        unassigned_players = [
            i for i in self.player_features.keys() if i not in assigned_players
        ]

        if len(unassigned_dets) == 1 and len(unassigned_players) == 1:
            det_idx = unassigned_dets[0]
            player_id = unassigned_players[0]
            player_assignments[det_idx] = player_id

            # Update feature
            if det_features[det_idx] is not None:
                self.player_features[player_id] = det_features[det_idx]

            # Record a re-ID switch
            self.metrics["reid_switches"] += 1

        return player_assignments

    def update_kalman_filter(self, player_id, measurement_pixel):
        """
        Update Kalman filter for a player with new measurement, applying constraints in real-world coords

        Args:
            player_id: ID of the player
            measurement_pixel: [x, y] position in pixel coordinates

        Returns:
            Updated Kalman filter
        """
        if player_id not in self.kalman_filters:
            self.kalman_filters[player_id] = self._init_kalman_filter()

        kf = self.kalman_filters[player_id]

        # Convert pixel measurement to real-world coordinates for more stable filtering
        real_measurement = None
        if self.homography is not None:
            try:
                real_pos = self.apply_homography(measurement_pixel)
                if real_pos is not None:
                    real_measurement = real_pos
            except:
                pass

        # If we can use real-world coordinates for filtering
        if real_measurement is not None:
            # If this is the first measurement for this player, initialize the filter
            if not hasattr(kf, "using_real_coords") or not kf.using_real_coords:
                # Reinitialize the filter for real-world coordinates
                kf = self._init_kalman_filter()
                kf.using_real_coords = True
                # Set initial state
                kf.x = np.array([real_measurement[0], real_measurement[1], 0, 0])

            # Predict and update in real-world coordinates
            kf.predict()
            kf.update(real_measurement)

            # Apply court constraints in real-world coordinates
            # Standard squash court dimensions in meters
            min_x_real = 0.0
            max_x_real = 6.4
            min_y_real = 0.0
            max_y_real = 9.75

            # Add margin
            margin_x_real = 1.0
            margin_y_real = 1.0

            # Check if prediction is outside court
            if (
                kf.x[0] < min_x_real - margin_x_real
                or kf.x[0] > max_x_real + margin_x_real
                or kf.x[1] < min_y_real - margin_y_real
                or kf.x[1] > max_y_real + margin_y_real
            ):

                # Record violation
                if player_id in self.metrics["court_violations"]:
                    self.metrics["court_violations"][player_id] += 1

                # Clip position to court boundaries (with margin)
                kf.x[0] = np.clip(
                    kf.x[0], min_x_real - margin_x_real, max_x_real + margin_x_real
                )
                kf.x[1] = np.clip(
                    kf.x[1], min_y_real - margin_y_real, max_y_real + margin_y_real
                )

                # Dampen velocity
                kf.x[2] *= 0.7  # x velocity
                kf.x[3] *= 0.7  # y velocity

            # Constrain velocity based on typical player movement speeds in squash
            # Max speed estimate: ~8 m/s for professional players
            max_velocity_real = 8.0  # meters per second
            max_velocity_per_frame = max_velocity_real / 30.0  # assuming 30 fps
            kf.x[2] = np.clip(kf.x[2], -max_velocity_per_frame, max_velocity_per_frame)
            kf.x[3] = np.clip(kf.x[3], -max_velocity_per_frame, max_velocity_per_frame)

            # Convert state back to pixel coordinates for display
            # Note: This is an approximation, true inverse homography would be better
            # but we're just using this for visualization, not tracking
            inverse_pt = None
            try:
                # Using inverse homography matrix if available
                if hasattr(self, "inv_homography") and self.inv_homography is not None:
                    inverse_pt = cv2.perspectiveTransform(
                        np.array([[[kf.x[0], kf.x[1]]]], dtype=np.float32),
                        self.inv_homography,
                    )[0][0]
            except:
                pass

            return kf

        else:
            # Fall back to pixel-based filtering if we can't use real-world coords
            kf.using_real_coords = False
            kf.predict()
            kf.update(measurement_pixel)

            # Apply simple constraints on pixel coordinates
            # These could be fine-tuned based on your video
            frame_width = 1920  # Assumed frame width
            frame_height = 1080  # Assumed frame height

            # Clip position to frame
            kf.x[0] = np.clip(kf.x[0], 0, frame_width)
            kf.x[1] = np.clip(kf.x[1], 0, frame_height)

            # Constrain velocity
            max_pixel_velocity = 30  # pixels per frame
            kf.x[2] = np.clip(kf.x[2], -max_pixel_velocity, max_pixel_velocity)
            kf.x[3] = np.clip(kf.x[3], -max_pixel_velocity, max_pixel_velocity)

            return kf

    def get_detections(self, frame):
        """Get player detections using YOLO"""
        results = self.model(frame)[0]
        detections = []

        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == self.player_class_id:
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                detections.append(
                    ([x, y, w, h], conf, self.config["player_class_name"])
                )

        return detections

    def process_frame(self, frame):
        """Process a video frame and track players"""
        import time

        start_time = time.time()

        # Get detections
        detections = self.get_detections(frame)

        # Apply court constraints using real-world coordinates
        detections = self.apply_court_constraints(detections)

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)

        # Assign detections to players
        player_assignments = self.assign_players(detections, frame)

        # Initialize return values
        player_pixel_coordinates = {}

        # Process each assigned detection
        for det_idx, player_id in player_assignments.items():
            bbox, conf, _ = detections[det_idx]
            x, y, w, h = bbox

            # Calculate center and feet positions
            center_x, center_y = x + w / 2, y + h / 2
            feet_x, feet_y = center_x, y + h

            # Update Kalman filter with FEET position instead of center
            kf = self.update_kalman_filter(player_id, [feet_x, feet_y])

            # Store position based on whether we're using real or pixel coordinates
            if hasattr(kf, "using_real_coords") and kf.using_real_coords:
                # We're tracking in real-world coordinates
                real_x, real_y = kf.x[0], kf.x[1]

                # Save real-world position directly
                if player_id not in self.player_real_positions:
                    self.player_real_positions[player_id] = []
                self.player_real_positions[player_id].append((real_x, real_y))

                # Convert back to pixel coordinates for display
                pixel_pos = None
                try:
                    if (
                        hasattr(self, "inv_homography")
                        and self.inv_homography is not None
                    ):
                        pixel_pos = cv2.perspectiveTransform(
                            np.array([[[real_x, real_y]]]), self.inv_homography
                        )[0][0]
                except:
                    # Fallback if inverse transformation fails
                    pixel_pos = (center_x, center_y)

                # When tracking in real-world coordinates, update this section
                if pixel_pos is not None:
                    # Store position for tracking history
                    if player_id not in self.player_positions:
                        self.player_positions[player_id] = deque(
                            maxlen=self.max_history
                        )
                    self.player_positions[player_id].append(pixel_pos)

                    # Estimate new bounding box position (adjusted to place feet at pixel_pos)
                    px, py = pixel_pos
                    new_x = px - w / 2
                    new_y = py - h  # This places the bottom of bbox at feet position
                    player_pixel_coordinates[player_id] = (
                        new_x,
                        new_y,
                        new_x + w,
                        new_y + h,
                    )

                else:
                    # Fallback to original bbox if transformation fails
                    player_pixel_coordinates[player_id] = (x, y, x + w, y + h)
            else:
                # We're tracking in pixel coordinates
                x_smooth, y_smooth = kf.x[0], kf.x[1]

                # Store pixel position
                if player_id not in self.player_positions:
                    self.player_positions[player_id] = deque(maxlen=self.max_history)
                self.player_positions[player_id].append((x_smooth, y_smooth))

                # Convert to real-world coordinates
                feet_pos = (x_smooth, y_smooth + h / 2)  # Bottom center
                real_xy = self.apply_homography(feet_pos)
                if real_xy is not None:
                    if player_id not in self.player_real_positions:
                        self.player_real_positions[player_id] = []
                    self.player_real_positions[player_id].append(real_xy)

                # Store pixel coordinates with smoothed position
                # In the else clause for pixel coordinate tracking
                # Store pixel coordinates with smoothed position based on feet
                new_x = x_smooth - w / 2  # Center horizontally on feet
                new_y = y_smooth - h  # Position top of box above feet
                player_pixel_coordinates[player_id] = (
                    new_x,
                    new_y,
                    new_x + w,
                    new_y + h,
                )

            # Update metrics
            if player_id not in self.metrics["player_confidence"]:
                self.metrics["player_confidence"][player_id] = []
                self.metrics["detection_count"][player_id] = 0
                self.metrics["court_violations"][player_id] = 0

            self.metrics["player_confidence"][player_id].append(conf)
            self.metrics["detection_count"][player_id] += 1

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time + 1e-6)
        self.metrics["tracking_fps"].append(fps)

        return player_pixel_coordinates, self.player_real_positions

    def get_all_positions(self):
        """Return all player positions tracked so far"""
        return self.player_real_positions
