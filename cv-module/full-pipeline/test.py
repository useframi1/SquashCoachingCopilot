from court_detection_pipeline import CourtCalibrator
import cv2
import numpy as np

# Initialize the calibrator
calibrator = CourtCalibrator()

# Load a video
cap = cv2.VideoCapture("video-5.mp4")

# Read the first frame
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read video frame")

# Detect keypoints and compute homography matrices
homographies, keypoints = calibrator.process_frame(frame)

# Draw keypoints on the frame
for class_name, kp_array in keypoints.items():
    for i, (x, y) in enumerate(kp_array):
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"{class_name}_{i}",
            (int(x) + 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

# Display the frame with keypoints
cv2.imshow("Court Keypoints", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get specific homography matrix
H_tbox = calibrator.get_homography("t-boxes")
H_wall = calibrator.get_homography("wall")

# Transform pixel coordinates to real-world coordinates
pixel_point = np.array([[(600, 500)]], dtype=np.float32)  # [x, y]
real_point = cv2.perspectiveTransform(pixel_point, H_tbox)
print("Real-world coordinates:", real_point[0][0])

cap.release()
