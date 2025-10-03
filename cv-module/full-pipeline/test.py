import cv2
from player_tracking_pipeline import PlayerTracker

tracker = PlayerTracker()
cap = cv2.VideoCapture("video-5.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = tracker.process_frame(frame)

    # Draw bounding boxes and positions
    for player_id in [1, 2]:
        if results[player_id]["bbox"]:
            x1, y1, x2, y2 = map(int, results[player_id]["bbox"])
            color = (0, 255, 0) if player_id == 1 else (255, 0, 0)

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw position
            pos = results[player_id]["position"]
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, color, -1)

            # Draw label
            label = f"P{player_id}: {results[player_id]['confidence']:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    cv2.imshow("Player Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
