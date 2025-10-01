import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from config import CONFIG


class PlayerTracker:
    """
    A clean, modular tracker for identifying and tracking two squash players.
    Returns the bottom-center position of each player's bounding box.
    """

    def __init__(self):
        """
        Initialize the player tracker.

        Args:
            max_history: Maximum number of historical positions to maintain
            reid_threshold: Threshold for re-identification matching (lower = stricter)
        """
        self.max_history = CONFIG["tracker"]["max_history"]
        self.reid_threshold = CONFIG["tracker"]["reid_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Player state
        self.player_positions = {
            1: deque(maxlen=self.max_history),
            2: deque(maxlen=self.max_history),
        }
        self.player_features = {1: None, 2: None}
        self.players_initialized = False

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Load YOLO detector and ResNet50 re-identification model."""
        print(f"Loading models on device: {self.device}")
        self.detector = YOLO(CONFIG["models"]["yolo_model"], verbose=False)

        # Load ResNet50 and modify on CPU first, then move to device
        self.reid_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.reid_model.fc.in_features
        feature_size = CONFIG["models"]["reid_feature_size"]
        self.reid_model.fc = torch.nn.Linear(in_features, feature_size)

        # Move entire model to device and set to eval mode
        self.reid_model = self.reid_model.to(self.device)
        self.reid_model.eval()
        print("Models loaded successfully")

    def track_frame(self, frame):
        """
        Process a single frame and return player tracking information.

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict: {
                1: {
                    "position": (x, y) or None,  # Player 1 bottom-center position
                    "bbox": [x1, y1, x2, y2] or None,  # Player 1 bounding box
                    "confidence": float or None  # Detection confidence
                },
                2: {
                    "position": (x, y) or None,  # Player 2 bottom-center position
                    "bbox": [x1, y1, x2, y2] or None,  # Player 2 bounding box
                    "confidence": float or None  # Detection confidence
                }
            }
        """
        frame_width = frame.shape[1]

        # Detect people in frame
        detections = self._detect_people(frame)

        # Assign detections to player IDs
        assignments = self._assign_detections(detections, frame, frame_width)

        # Initialize results
        results = {
            1: {"position": None, "bbox": None, "confidence": None},
            2: {"position": None, "bbox": None, "confidence": None},
        }

        for det_idx, player_id in assignments.items():
            bbox = detections[det_idx][0:4]
            confidence = detections[det_idx][4]

            # Calculate bottom-center position
            center_x = (bbox[0] + bbox[2]) / 2
            bottom_y = bbox[3]
            position = (center_x, bottom_y)

            # Store all tracking information
            results[player_id] = {
                "position": position,
                "bbox": bbox.tolist(),
                "confidence": float(confidence),
            }

            # Update position history
            self.player_positions[player_id].append(position)

        return results

    def _detect_people(self, frame):
        """
        Detect people in the frame using YOLO.

        Returns:
            list: Top 2 person detections sorted by confidence
        """
        results = self.detector(frame, verbose=False)[0]
        detections = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        detections = np.hstack([detections, confidences[:, None], classes[:, None]])

        # Filter for person class (class 0) and get top 2
        person_dets = [det for det in detections if det[5] == 0]
        person_dets = sorted(person_dets, key=lambda x: x[4], reverse=True)[:2]

        return person_dets

    def _assign_detections(self, detections, frame, frame_width):
        """
        Assign detections to player IDs using re-identification and position.

        Returns:
            dict: {detection_index: player_id}
        """
        if len(detections) == 0:
            return {}

        # Initialize players on first frame
        if not self.players_initialized:
            return self._initialize_players(detections, frame)

        # Extract features for current detections
        det_features = [
            self._extract_features(frame, det[0:4]) for det in detections[:2]
        ]

        # Reinitialize if both players lost
        if self.player_features[1] is None and self.player_features[2] is None:
            return self._initialize_players(detections, frame)

        # Calculate matching scores
        matching_scores = self._calculate_matching_scores(
            detections, det_features, frame_width
        )

        # Assign detections to players
        assignments = self._greedy_assignment(matching_scores, det_features)

        return assignments

    def _initialize_players(self, detections, frame):
        """
        Initialize players based on left/right positioning.

        Returns:
            dict: Initial player assignments
        """
        if len(detections) < 2:
            print("Warning: Less than 2 players detected")
            return {}

        # Extract features
        det_features = [
            self._extract_features(frame, det[0:4]) for det in detections[:2]
        ]

        # Sort by x-coordinate (left to right)
        detections_with_idx = [(i, det) for i, det in enumerate(detections[:2])]
        detections_with_idx.sort(key=lambda x: (x[1][0] + x[1][2]) / 2)

        # Assign: leftmost = Player 1, rightmost = Player 2
        assignments = {}
        for player_id, (det_idx, _) in enumerate(detections_with_idx, 1):
            assignments[det_idx] = player_id
            self.player_features[player_id] = det_features[det_idx]

        self.players_initialized = True
        return assignments

    def _calculate_matching_scores(self, detections, det_features, frame_width):
        """
        Calculate matching scores between detections and tracked players.

        Returns:
            dict: {(det_idx, player_id): score}
        """
        matching_scores = {}

        for i, det in enumerate(detections[:2]):
            bbox = det[0:4]
            center = ((bbox[0] + bbox[2]) / 2, bbox[3])

            for player_id in [1, 2]:
                if self.player_features[player_id] is None:
                    continue

                # Re-identification score
                reid_score = self._feature_distance(
                    det_features[i], self.player_features[player_id]
                )

                # Position score
                pos_score = 0
                if len(self.player_positions[player_id]) > 0:
                    last_pos = self.player_positions[player_id][-1]
                    distance = np.sqrt(
                        (center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2
                    )
                    pos_score = distance / frame_width

                # Combined score (weighted)
                reid_weight = CONFIG["tracker"]["reid_weight"]
                pos_weight = CONFIG["tracker"]["position_weight"]
                score = reid_score * reid_weight + pos_score * pos_weight
                matching_scores[(i, player_id)] = score
        return matching_scores

    def _greedy_assignment(self, matching_scores, det_features):
        """
        Perform greedy assignment of detections to players.

        Returns:
            dict: {det_idx: player_id}
        """
        assignments = {}
        assigned_players = set()

        # Assign based on best scores
        for (det_idx, player_id), score in sorted(
            matching_scores.items(), key=lambda x: x[1]
        ):
            if det_idx in assignments or player_id in assigned_players:
                continue
            if score < self.reid_threshold:
                assignments[det_idx] = player_id
                assigned_players.add(player_id)

                # Update player features with exponential moving average
                if det_features[det_idx] is not None:
                    self.player_features[player_id] = (
                        0.5 * self.player_features[player_id]
                        + 0.5 * det_features[det_idx]
                    )

        # Handle remaining unassigned detections
        unassigned_dets = [
            i for i in range(min(2, len(det_features))) if i not in assignments
        ]
        unassigned_players = [i for i in [1, 2] if i not in assigned_players]

        if len(unassigned_dets) == 1 and len(unassigned_players) == 1:
            det_idx = unassigned_dets[0]
            player_id = unassigned_players[0]
            assignments[det_idx] = player_id
            self.player_features[player_id] = det_features[det_idx]

        return assignments

    def _extract_features(self, frame, bbox):
        """
        Extract re-identification features from a bounding box.

        Returns:
            numpy.ndarray: Feature vector or None if invalid bbox
        """
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)

        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None

        # Prepare image for ResNet50
        input_size = tuple(CONFIG["models"]["reid_input_size"])
        person_img = cv2.resize(person_img, input_size)
        person_tensor = torch.from_numpy(person_img).permute(2, 0, 1).float() / 255.0
        person_tensor = person_tensor.unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.reid_model(person_tensor)

        return features.squeeze().cpu().numpy()

    def _feature_distance(self, feat1, feat2):
        """
        Calculate cosine distance between two feature vectors.

        Returns:
            float: Distance (0 = identical, 2 = opposite)
        """
        if feat1 is None or feat2 is None:
            return float("inf")

        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)

        return 1 - np.dot(feat1, feat2)

    def reset(self):
        """Reset tracker state (useful for processing new videos)."""
        self.player_positions = {
            1: deque(maxlen=self.max_history),
            2: deque(maxlen=self.max_history),
        }
        self.player_features = {1: None, 2: None}
        self.players_initialized = False


# ========== USAGE EXAMPLE ==========

if __name__ == "__main__":
    # Initialize tracker
    tracker = PlayerTracker()

    # Open video
    cap = cv2.VideoCapture(CONFIG["paths"]["test_video"])

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Track players in frame
        results = tracker.track_frame(frame)

        frame_count += 1

        # Optional: visualize (for debugging)
        if results[1]["position"]:
            pos = results[1]["position"]
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)
        if results[2]["position"]:
            pos = results[2]["position"]
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, 0, 255), -1)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
