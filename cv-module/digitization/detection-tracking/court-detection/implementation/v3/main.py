from inference import get_model
import cv2
import numpy as np


def get_quadrilateral_corners(points, epsilon_factor=0.02):
    """
    Approximate segmentation to exactly 4 corners using polygon approximation

    Args:
        points: List of Point objects from Roboflow prediction
        epsilon_factor: Controls approximation accuracy (lower = more precise to original shape)

    Returns:
        numpy array of shape (4, 2) containing [x, y] coordinates of corners
    """
    # Convert to numpy array
    coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)

    # Calculate perimeter
    perimeter = cv2.arcLength(coords, True)

    # Approximate polygon - try to get exactly 4 points
    # Start with a reasonable epsilon
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(coords, epsilon, True)

    # If we don't get 4 points, adjust epsilon
    attempts = 0
    while len(approx) != 4 and attempts < 20:
        if len(approx) > 4:
            # Too many points, increase epsilon (more aggressive approximation)
            epsilon_factor *= 1.2
        else:
            # Too few points, decrease epsilon (less aggressive approximation)
            epsilon_factor *= 0.8

        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(coords, epsilon, True)
        attempts += 1

    # If we still don't have 4 points, fall back to convex hull or rotated rect
    if len(approx) != 4:
        # Try convex hull first
        hull = cv2.convexHull(coords)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)

        # If still not 4 points, use rotated rectangle as fallback
        if len(approx) != 4:
            rect = cv2.minAreaRect(coords)
            approx = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.float32)

    return approx.reshape(4, 2).astype(np.int32)


def order_corners(corners):
    """
    Order corners as: top-left, top-right, bottom-right, bottom-left
    """
    # Sort by y-coordinate
    corners = sorted(corners, key=lambda x: x[1])

    # Top two points (sorted left to right)
    top = sorted(corners[:2], key=lambda x: x[0])
    # Bottom two points (sorted left to right)
    bottom = sorted(corners[2:], key=lambda x: x[0])

    return np.array([top[0], top[1], bottom[1], bottom[0]])


# Initialize your model
model = get_model(model_id="court-detection-pdgxs/1", api_key="8ztE8wKb3wAhHf8BARWy")

# Run inference
image_path = "image-1.png"
results = model.infer(image_path)

# Load image for visualization
image = cv2.imread(image_path)

# Access predictions
predictions = results[0].predictions

# Process each prediction
for prediction in predictions:
    # Filter for squares and tin
    if prediction.class_name in [
        "tin",
        "left-square",
        "right-square",
        "front-wall-down",
    ]:
        # Get the 4 corners
        corners = get_quadrilateral_corners(prediction.points, epsilon_factor=0.02)

        # Order corners consistently
        ordered_corners = order_corners(corners)

        print(f"\nClass: {prediction.class_name}")
        print(f"Confidence: {prediction.confidence:.2f}")
        print(f"4 Corners (TL, TR, BR, BL):\n{ordered_corners}")

        # Draw the quadrilateral on the image
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)

        # Label each corner with ordered numbering
        for i, corner in enumerate(ordered_corners):
            cv2.circle(image, tuple(corner), 7, (255, 0, 0), -1)
            cv2.putText(
                image,
                str(i),
                (corner[0] + 10, corner[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Add class label
        center_x = int(np.mean(ordered_corners[:, 0]))
        center_y = int(np.mean(ordered_corners[:, 1]))
        cv2.putText(
            image,
            prediction.class_name,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

cv2.imshow("Court Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
