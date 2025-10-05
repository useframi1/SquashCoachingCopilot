import cv2
import json
import os
from datetime import timedelta
from ultralytics import YOLO
import numpy as np


class SquashAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.annotations = []

    def add_annotation(self, event_type):
        annotation = {
            "frame": self.current_frame,
            "time": round(self.current_frame / self.fps, 3),
            "event": event_type,
        }
        self.annotations.append(annotation)
        print(f"{event_type} marked at frame {self.current_frame}")
        return annotation


def run_integrated_pipeline(video_path, output_path):
    # Load YOLOv11 pose model
    model = YOLO("yolo11n-pose.pt")

    # COCO keypoints indices for relevant body parts
    RELEVANT_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    RELEVANT_NAMES = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    # Create annotator instance
    annotator = SquashAnnotator(video_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Combined output data
    integrated_data = []

    # Control variables
    playing = False
    selected_player_id = None
    id_input_buffer = ""
    id_input_mode = False
    frame_idx = 0
    current_frame_data = None
    temp_event = None

    # Window setup
    cv2.namedWindow("Pose Tracking & Annotation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Tracking & Annotation", 1280, 720)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    frame_idx = 1
    annotator.current_frame = frame_idx

    # Window setup
    cv2.namedWindow("Pose Tracking & Annotation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Tracking & Annotation", 1280, 720)

    # Control variables
    playing = False
    selected_player_id = None
    id_input_buffer = ""
    id_input_mode = False

    while cap.isOpened():
        # Only read a new frame if playing or at the start
        if playing:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            annotator.current_frame = frame_idx
            temp_event = None  # Reset temporary event when advancing frames

        # Process frame with YOLO
        results = model.track(source=frame, persist=True, conf=0.6, verbose=False)
        result = results[0]

        keypoints_data = result.keypoints
        ids = (
            result.boxes.id
            if result.boxes is not None and hasattr(result.boxes, "id")
            else None
        )

        # Create or update frame data for the current frame
        if current_frame_data is None or current_frame_data["frame"] != frame_idx:
            current_frame_data = {"frame": frame_idx, "time": round(frame_idx / fps, 3)}

            # Add any temporary event if it exists
            if temp_event:
                current_frame_data["event"] = temp_event

        # Process keypoints for selected player
        if keypoints_data is not None and selected_player_id is not None:
            for i, person_keypoints in enumerate(keypoints_data.data):
                if len(person_keypoints) <= max(RELEVANT_IDX):
                    continue  # Skip if not enough keypoints

                current_id = None
                if ids is not None and i < len(ids):
                    current_id = int(ids[i].item())

                # Only process the selected player
                if current_id == selected_player_id:
                    keypoints = {}

                    for j, idx in enumerate(RELEVANT_IDX):
                        kp = person_keypoints[idx].cpu().numpy().tolist()
                        keypoints[RELEVANT_NAMES[j]] = {
                            "x": kp[0],
                            "y": kp[1],
                            "confidence": kp[2],
                        }

                    current_frame_data["keypoints"] = keypoints
                    current_frame_data["player_id"] = current_id

                    # If this frame isn't already in integrated_data, add it
                    frame_already_added = False
                    for i, frame_data in enumerate(integrated_data):
                        if frame_data["frame"] == frame_idx:
                            integrated_data[i] = current_frame_data
                            frame_already_added = True
                            break

                    if not frame_already_added:
                        integrated_data.append(current_frame_data)
                    break

        # Display information
        annotated = result.plot()

        # Add UI information
        time_str = str(timedelta(seconds=frame_idx / fps))
        cv2.putText(
            annotated,
            f"Frame: {frame_idx}/{total_frames} Time: {time_str}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if id_input_mode:
            cv2.putText(
                annotated,
                f"Enter player ID: {id_input_buffer}_",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
        elif selected_player_id is not None:
            cv2.putText(
                annotated,
                f"Selected Player ID: {selected_player_id}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                annotated,
                "No player selected (press P to select player)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if temp_event:
            cv2.putText(
                annotated,
                f"Stroke: {temp_event}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        instructions = [
            "Space: Play/Pause",
            "P: Enter player ID mode",
            "0: Track all players (clear selection)",
            "'F': Forehand, 'B': Backhand",
            "'Q': Quit & Save",
        ]

        for i, text in enumerate(instructions):
            cv2.putText(
                annotated,
                text,
                (10, 120 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Show the frame
        cv2.imshow("Pose Tracking & Annotation", annotated)

        # Process keyboard input
        if playing:
            key = cv2.waitKey(30) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        if id_input_mode:
            if ord("0") <= key <= ord("9"):
                id_input_buffer += chr(key)
            elif key == 13 or key == 10:  # Enter key
                if id_input_buffer:
                    selected_player_id = int(id_input_buffer)
                    print(f"Selected player ID: {selected_player_id}")
                id_input_buffer = ""
                id_input_mode = False
            elif key == 27 or key == ord("q"):  # Escape key or q
                id_input_buffer = ""
                id_input_mode = False
            elif key == 8:  # Backspace
                if id_input_buffer:
                    id_input_buffer = id_input_buffer[:-1]
        else:
            if key == ord("q"):
                break
            elif key == ord(" "):
                playing = not playing
            elif key == ord("f"):
                temp_event = "forehand"
                annotation = annotator.add_annotation("forehand")
                if current_frame_data:
                    current_frame_data["event"] = "forehand"
            elif key == ord("b"):
                temp_event = "backhand"
                annotation = annotator.add_annotation("backhand")
                if current_frame_data:
                    current_frame_data["event"] = "backhand"
            elif key == ord("p"):
                id_input_mode = True
                id_input_buffer = ""
            elif key == ord("0"):
                selected_player_id = None
                print("No player selected")

    # Save integrated data to a single JSON file (filter out frames with no player_id or keypoints)
    filtered_data = [
        frame
        for frame in integrated_data
        if "player_id" in frame and "keypoints" in frame
    ]
    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Saved {len(filtered_data)} frames of data to {output_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # File paths
    video_path = "cv-module/digitization/event-recognition/stroke-detection/implementation/Videos/video-4.mp4"
    output_path = "cv-module/digitization/event-recognition/stroke-detection/implementation/annotated_jsons/video_4_annotated.json"

    # Run the integrated pipeline
    run_integrated_pipeline(video_path, output_path)
