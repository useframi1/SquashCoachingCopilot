import cv2

from rally_state_pipeline import RallyStateDetector
from player_tracker import PlayerTracker
from court_calibrator import CourtCalibrator


def main():
    rally_state_detector = RallyStateDetector()
    court_calibrator = CourtCalibrator()
    player_tracker = None

    cap = cv2.VideoCapture("video-5.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    homography_matrix = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute homography matrix only for the first frame
        if frame_count == 0:
            homography_matrix = court_calibrator.compute_homography(frame)
            if homography_matrix is not None:
                player_tracker = PlayerTracker(homography_matrix)
            else:
                print("Error: Could not compute homography matrix.")
                break

        # Get player coordinates
        if player_tracker is not None:
            player_coords = player_tracker.process_frame(frame)

            # Get rally state
            rally_state = rally_state_detector.process_frame(player_coords)

            # Display rally state on frame
            cv2.putText(
                frame,
                f"Rally State: {rally_state}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Display the frame
        cv2.imshow("Video", frame)

        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
