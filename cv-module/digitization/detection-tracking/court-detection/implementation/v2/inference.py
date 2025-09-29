# detect_first_frame.py
import cv2
from ultralytics import YOLO

# Configuration
VIDEO_PATH = "video-3.mp4"
MODEL_PATH = "runs/pose/court-detection/weights/best.pt"  # Your trained model

# Load model
print("Loading model...")
model = YOLO(MODEL_PATH)

# Open video
print(f"Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# Read first frame
ret, frame = cap.read()
cap.release()  # Close video immediately

if not ret:
    print("Error: Could not read first frame")
    exit()

print("Processing first frame...")

# Run inference on first frame
results = model(frame)

# Define color and drawing parameters
keypoint_color = (0, 0, 255)  # Red
radius = 5
thickness = -1  # Filled circle

# Draw keypoints
for result in results:
    kps = result.keypoints
    print(kps)
    if kps is not None and hasattr(kps, "data"):
        keypoints_array = kps.data.cpu().numpy()  # shape: (n, num_keypoints, 3)
        for person_keypoints in keypoints_array:
            for x, y, conf in person_keypoints:
                if conf > 0.5:  # Only show confident keypoints
                    cv2.circle(
                        frame, (int(x), int(y)), radius, keypoint_color, thickness
                    )

# Show image
cv2.imshow("Keypoints", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
