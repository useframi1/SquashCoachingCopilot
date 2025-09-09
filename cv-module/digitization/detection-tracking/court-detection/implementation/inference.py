from ultralytics import YOLO
import cv2

# Load model
model = YOLO("runs/train/squash_court_tboxes/weights/best.pt")

# Load image
image_path = "yolo_dataset/train/images/029AwqdH868-000000.jpg"
image = cv2.imread(image_path)

# Run inference
results = model(image)

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
                        image, (int(x), int(y)), radius, keypoint_color, thickness
                    )

# Show image
cv2.imshow("Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
