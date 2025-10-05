import cv2
import json
from ultralytics import YOLO

# Load YOLOv11 pose model
model = YOLO("yolo11n-pose.pt")

# Video path
video_path = "clip.mp4"
output_json_path = "clip_keypoints.json"

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

# Open video
cap = cv2.VideoCapture(video_path)
all_keypoints = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    results = model.track(source=frame, persist=True, conf=0.6, verbose=False)
    result = results[0]
    frame_people = []

    keypoints_data = result.keypoints
    ids = (
        result.boxes.id
        if result.boxes is not None and hasattr(result.boxes, "id")
        else None
    )

    if keypoints_data is not None:
        for i, person_keypoints in enumerate(keypoints_data.data):
            if len(person_keypoints) <= max(RELEVANT_IDX):
                continue  # Skip if not enough keypoints

            person_data = {}

            # Add tracking ID if available
            if ids is not None and i < len(ids):
                person_data["id"] = int(ids[i].item())

            for j, idx in enumerate(RELEVANT_IDX):
                kp = person_keypoints[idx].cpu().numpy().tolist()
                person_data[RELEVANT_NAMES[j]] = {
                    "x": kp[0],
                    "y": kp[1],
                    "confidence": kp[2],
                }

            frame_people.append(person_data)

    all_keypoints.append({"frame": frame_idx, "people": frame_people})

    # Show the frame
    annotated = result.plot()
    cv2.imshow("Pose Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save keypoints with IDs
with open(output_json_path, "w") as f:
    json.dump(all_keypoints, f, indent=4)

print(f"Saved {frame_idx} frames of relevant keypoints to {output_json_path}")
