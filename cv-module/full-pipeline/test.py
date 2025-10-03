from ball_detection_pipeline import BallTracker
import cv2

# Initialize the tracker
tracker = BallTracker()
frame_count = 0
# Process video frames
cap = cv2.VideoCapture("video-5.mp4")
while cap.isOpened() and frame_count < 100:
    ret, frame = cap.read()
    if not ret:
        break

    # Get ball coordinates
    x, y = tracker.process_frame(frame)

    if x is not None and y is not None:
        # Draw ball position
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Ball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
