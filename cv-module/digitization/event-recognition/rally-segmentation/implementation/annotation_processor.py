import os
import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm  # For progress bar
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from itertools import groupby
from scipy.spatial import distance

from court_calibrator import CourtCalibrator
from model import BallTrackerNet
from general import postprocess


class SquashAnnotationProcessor:
    def __init__(
        self,
        annotation_file: str,
        player_detection_model_path: str,
        ball_detection_model_path: str,
        keypoint_detection_model_path: str,
        output_file: str = "annotations_with_detections.json",
    ):
        """
        Initialize the SquashAnnotationProcessor.

        Args:
            annotation_file: Path to the annotations.json file
            player_detection_model_path: Path to the player detection model
            ball_detection_model_path: Path to the ball detection model
            keypoint_detection_model_path: Path to the keypoint detection model
            output_file: Path where to save the output JSON file
        """
        self.annotation_file = annotation_file
        self.output_file = output_file
        self.annotations = self._load_annotations()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load the detection models
        self.player_detector = self._load_player_detection_model(
            player_detection_model_path
        )
        self.ball_detector = self._load_ball_detection_model(ball_detection_model_path)
        self.homography_estimator = self._load_homography_estimator_model(
            keypoint_detection_model_path
        )

    def _load_annotations(self) -> Dict:
        """Load the annotations from the JSON file."""
        with open(self.annotation_file, "r") as f:
            return json.load(f)

    def _apply_homography(self, pt, homography):
        transformed_point = cv2.perspectiveTransform(
            np.array([[pt]], dtype=np.float32), homography
        )
        return transformed_point[0][0]

    def _remove_outliers(self, ball_track, dists, max_dist=100):
        """Remove outliers from model prediction
        :params
            ball_track: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
            max_dist: maximum distance between two neighbouring ball points
        :return
            ball_track: list of ball points
        """
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        print(f"Number of outliers: {len(outliers)}")
        for i in outliers:
            if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i - 1] == -1:
                ball_track[i - 1] = (None, None)
        return ball_track

    def _split_track(self, ball_track, max_gap=4, max_dist_gap=80, min_track=5):
        """Split ball track into several subtracks in each of which we will perform
        ball interpolation.
        :params
            ball_track: list of detected ball points
            max_gap: maximun number of coherent None values for interpolation
            max_dist_gap: maximum distance at which neighboring points remain in one subtrack
            min_track: minimum number of frames in each subtrack
        :return
            result: list of subtrack indexes
        """
        list_det = [0 if x[0] else 1 for x in ball_track]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

        cursor = 0
        min_value = 0
        result = []
        for i, (k, l) in enumerate(groups):
            if (k == 1) & (i > 0) & (i < len(groups) - 1):
                dist = distance.euclidean(
                    ball_track[cursor - 1], ball_track[cursor + l]
                )
                if (l >= max_gap) | (dist / l > max_dist_gap):
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                        min_value = cursor + l - 1
            cursor += l
        if len(list_det) - min_value > min_track:
            result.append([min_value, len(list_det)])
        return result

    def _interpolation(self, coords):
        """Run ball interpolation in one subtrack
        :params
            coords: list of ball coordinates of one subtrack
        :return
            track: list of interpolated ball coordinates of one subtrack
        """

        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

        nons, yy = nan_helper(x)
        x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
        nans, xx = nan_helper(y)
        y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

        track = [*zip(x, y)]
        return track

    def _infer_ball_detection(
        self, frames: np.ndarray, ball_track
    ) -> Optional[Dict[str, Any]]:
        """
        Run ball detection on a frame.
        Replace with your actual ball detection code.

        Returns:
            Ball detection with position and confidence, or None if not detected
        """
        height = 360
        width = 640

        img = cv2.resize(frames[2], (width, height))
        img_prev = cv2.resize(frames[1], (width, height))
        img_preprev = cv2.resize(frames[0], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = self.ball_detector(torch.from_numpy(inp).float().to(self.device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)

        if (x_pred or y_pred) and ball_track[-1][0]:
            dist = distance.euclidean([x_pred, y_pred], ball_track[-1])
            print([x_pred, y_pred], ball_track[-1])
            print(dist)
        else:
            dist = -1

        if x_pred and y_pred:
            return [int(x_pred), int(y_pred)], dist
        else:
            return [None, None], dist

    def _load_player_detection_model(self, model_path: str) -> Any:
        """
        Load the player detection model.
        """
        print(f"Loading player detection model from {model_path}")
        model = YOLO(model_path)
        return model

    def _load_ball_detection_model(self, model_path: str) -> Any:
        """
        Load the ball detection model.
        """
        print(f"Loading ball detection model from {model_path}")
        model = BallTrackerNet()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        return model

    def _load_homography_estimator_model(self, model_path: str) -> CourtCalibrator:
        """
        Load the keypoint detection model for court corners.
        """
        print(f"Loading keypoint detection model from {model_path}")
        court_calibrator = CourtCalibrator(model_path=model_path)
        return court_calibrator

    def _detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 player detection on a frame and return the bottom middle point of each bounding box.

        Args:
            frame: The video frame to process

        Returns:
            List of detected players with bounding boxes, confidence scores, and bottom middle point
        """
        # Run YOLOv8 detection
        results = self.player_detector(frame)[0]

        # Process the detection results
        bb_detections = []

        # For older versions of YOLOv8 or different format
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:  # person class
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                bb_detections.append(([x, y, w, h], conf, "player"))

        # Handle case if more than 2 players are detected
        # Sort by confidence and keep only the top 2
        if len(bb_detections) > 2:
            bb_detections.sort(key=lambda x: x[1], reverse=True)
            bb_detections = bb_detections[:2]

        player_detections = []

        # Update the tracker
        tracks = self.tracker.update_tracks(bb_detections, frame=frame)

        for track in tracks[:2]:
            if not track.is_confirmed():
                continue
            pid = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()

            foot = ((x1 + x2) / 2, y2)

            player_detections.append(
                {
                    "player_id": pid,
                    "bottom_middle": [int(foot[0]), int(foot[1])],
                }
            )

        return player_detections

    def _detect_ball(self, frames: np.ndarray, ball_track) -> Optional[Dict[str, Any]]:
        """
        Run ball detection on a frame.
        Replace with your actual ball detection code.

        Returns:
            Ball detection with position and confidence, or None if not detected
        """
        ball_track, dists = self._infer_ball_detection(frames, ball_track)

        return ball_track, dists

    def _detect_players_position(
        self, frame: np.ndarray, players: List
    ) -> Dict[str, Any]:
        """
        Run keypoint detection for the 4 corners of the T-boxes.

        Returns:
            List of actual players' position on the court.
        """
        H = self.homography_estimator.compute_homography(frame)
        if H is None:
            return []

        players_position = []
        for player in players:
            x, y = player["bottom_middle"]
            position = self._apply_homography(pt=(x, y), homography=H)
            players_position.append(position.tolist())

        return players_position

    def process_video_frames_efficiently(
        self, video_path: str, frame_range: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process multiple frames from a video efficiently by opening the video once.

        Args:
            video_path: Path to the video file
            frame_range: List of frame numbers to process

        Returns:
            Dict mapping frame numbers to detection results
        """
        # Sort frame numbers to process them in order
        frame_range = sorted(frame_range)
        if not frame_range:
            return {}

        results = {}
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return {
                frame: {"frame_number": frame, "players_position": [], "ball": None}
                for frame in frame_range
            }

        # Get the first frame to process
        current_frame = 0
        ball_detection_frames = []
        ball_track_all = []
        dists_all = []
        for idx, frame_num in enumerate(frame_range):
            # Seek to the required position
            if frame_num < current_frame:
                # If we need to go back, reopen the video
                cap.release()
                cap = cv2.VideoCapture(video_path)
                current_frame = 0

            # Skip frames until we reach the desired frame
            while current_frame < frame_num:
                ret = cap.grab()  # Just grab the frame without decoding
                if not ret:
                    break
                current_frame += 1

            # Read the actual frame we want to process
            ret, frame = cap.read()
            if ret:
                current_frame += 1
                ball_detection_frames.append(frame)
                if len(ball_detection_frames) % 3 == 0:
                    ball, dists = self._detect_ball(
                        ball_detection_frames, ball_track_all
                    )
                    dists_all.append(dists)
                    ball_detection_frames = ball_detection_frames[1:]
                else:
                    ball = [None, None]
                    dists_all.append(-1)

                print(ball)
                ball_track_all.append(ball)

                # Process the frame
                players = self._detect_players(frame)
                players_position = self._detect_players_position(frame, players)

                # Store the results
                results[idx] = {
                    "frame_number": frame_num,
                    "players_position": players_position,
                    "ball": ball,
                }
            else:
                print(f"Warning: Could not read frame {frame_num} from {video_path}")
                results[idx] = {
                    "frame_number": frame_num,
                    "players_position": [],
                    "ball": None,
                }

        ball_tracks_all = self._remove_outliers(ball_track_all, dists_all)
        subtracks = self._split_track(ball_track_all)
        for r in subtracks:
            ball_subtrack = ball_tracks_all[r[0] : r[1]]
            ball_subtrack = self._interpolation(ball_subtrack)
            ball_tracks_all[r[0] : r[1]] = ball_subtrack

        for i in range(len(results)):
            results[i]["ball"] = ball_tracks_all[i]

        cap.release()

        return results

    def process_annotations(self, videos_dir: str, save_progress: bool = True) -> None:
        """
        Process all annotations, run models on all frames between start_frame and end_frame.

        Args:
            videos_dir: Directory containing the video files
            save_progress: Whether to save progress after each video
        """
        enhanced_annotations = {}

        for video_file, annotations in self.annotations.items():
            print(f"Processing video: {video_file}")
            video_path = os.path.join(videos_dir, video_file)

            self.tracker = DeepSort(max_age=30)

            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"Warning: Video file {video_path} not found. Skipping.")
                enhanced_annotations[video_file] = annotations
                continue

            # Process each annotation
            enhanced_video_annotations = []

            for annotation in tqdm(
                annotations, desc=f"Processing annotations for {video_file}"
            ):
                # Get the start and end frames
                start_frame = annotation["start_frame"]
                end_frame = annotation["end_frame"]

                # Create a new annotation with the original data
                enhanced_annotation = annotation.copy()

                # Create the range of frames to process
                frames_to_analyze = list(range(start_frame, end_frame + 1))

                # Process all frames in the range efficiently
                print(
                    f"Processing frames {start_frame} to {end_frame} ({len(frames_to_analyze)} frames)"
                )
                frame_detections = self.process_video_frames_efficiently(
                    video_path, frames_to_analyze
                )

                # Add frame detections to the enhanced annotation
                enhanced_annotation["detections"] = frame_detections
                enhanced_video_annotations.append(enhanced_annotation)

            enhanced_annotations[video_file] = enhanced_video_annotations

            # Optionally save progress after each video
            if save_progress:
                with open(self.output_file, "w") as f:
                    json.dump(enhanced_annotations, f, indent=2)
                print(
                    f"Progress saved to {self.output_file} after processing {video_file}"
                )

        # Save the final enhanced annotations to the output file
        with open(self.output_file, "w") as f:
            json.dump(enhanced_annotations, f, indent=2)

        print(f"Enhanced annotations saved to {self.output_file}")

    def process_batch(self, videos_dir: str, batch_size: int = 5) -> None:
        """
        Process annotations in batches to manage memory efficiently.

        Args:
            videos_dir: Directory containing the video files
            batch_size: Number of annotations to process at once
        """
        enhanced_annotations = {}

        for video_file, annotations in self.annotations.items():
            print(f"Processing video: {video_file}")
            video_path = os.path.join(videos_dir, video_file)

            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"Warning: Video file {video_path} not found. Skipping.")
                enhanced_annotations[video_file] = annotations
                continue

            # Process annotations in batches
            enhanced_video_annotations = []
            for i in range(0, len(annotations), batch_size):
                batch = annotations[i : i + batch_size]
                print(
                    f"Processing batch {i//batch_size + 1}/{(len(annotations) + batch_size - 1)//batch_size}"
                )

                for annotation in tqdm(batch, desc=f"Processing annotations"):
                    # Get the start and end frames
                    start_frame = annotation["start_frame"]
                    end_frame = annotation["end_frame"]

                    # Create a new annotation with the original data
                    enhanced_annotation = annotation.copy()

                    # Create the range of frames to process
                    frames_to_analyze = list(range(start_frame, end_frame + 1))

                    # Process all frames in the range efficiently
                    frame_detections = self.process_video_frames_efficiently(
                        video_path, frames_to_analyze
                    )

                    # Add frame detections to the enhanced annotation
                    enhanced_annotation["detections"] = frame_detections
                    enhanced_video_annotations.append(enhanced_annotation)

                # Save progress after each batch
                enhanced_annotations[video_file] = enhanced_video_annotations
                with open(self.output_file, "w") as f:
                    json.dump(enhanced_annotations, f, indent=2)
                print(f"Progress saved after batch {i//batch_size + 1}")

        print(f"All processing complete. Results saved to {self.output_file}")


# Example usage
if __name__ == "__main__":
    processor = SquashAnnotationProcessor(
        annotation_file="annotations.json",
        player_detection_model_path="yolo11s.pt",
        ball_detection_model_path="models/ball_detector.pt",
        keypoint_detection_model_path="models/tbox_detector.pt",
        output_file="annotations_with_detections.json",
    )

    # Process all annotations with all frames and save the results
    processor.process_annotations(videos_dir="videos")

    # Alternatively, use batch processing for better memory management
    # processor.process_batch(videos_dir="videos", batch_size=5)
