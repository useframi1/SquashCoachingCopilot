from court_calibrator import CourtCalibrator
from player_tracker import PlayerTracker
import cv2
import json
import numpy as np


def is_rally_start(p1_pos, p2_pos, p1_intensity, p2_intensity):
    """Check if players are in rally start positions with specific conditions"""
    if p1_pos is None or p2_pos is None or p1_intensity is None or p2_intensity is None:
        return False

    # Both players must have low intensity
    LOW_INTENSITY_THRESHOLD = 0.025
    if p1_intensity > LOW_INTENSITY_THRESHOLD or p2_intensity > LOW_INTENSITY_THRESHOLD:
        return False

    # Both players must be before the 5.44m mark (closer to front wall)
    SERVICE_LINE = 5.44
    if p1_pos[1] < SERVICE_LINE or p2_pos[1] < SERVICE_LINE:
        return False

    # Players must be on different sides of the court (different x coordinates)
    # Court width is 6.4m, so center is at 3.2m
    COURT_CENTER_X = 3.2
    p1_left = p1_pos[0] < COURT_CENTER_X
    p2_left = p2_pos[0] < COURT_CENTER_X
    if p1_left == p2_left:  # Both on same side
        return False

    print(f"Debug: P1 pos: {p1_pos}, P2 pos: {p2_pos}")

    # At least one player should be in service box area (back portion before 5.44m)
    # Service boxes are roughly between 7.04m and 5.44m from front wall
    SERVICE_BOX_BACK = 7.04
    p1_in_box = (SERVICE_LINE <= p1_pos[1] <= SERVICE_BOX_BACK) and (
        0 <= p1_pos[0] <= 1.6 or 4.8 <= p1_pos[0] <= 6.4
    )
    p2_in_box = (SERVICE_LINE <= p2_pos[1] <= SERVICE_BOX_BACK) and (
        0 <= p2_pos[0] <= 1.6 or 4.8 <= p2_pos[0] <= 6.4
    )

    print(f"Debug: P1 in box: {p1_in_box}, P2 in box: {p2_in_box}")

    return p1_in_box or p2_in_box


def is_rally_active(p1_pos, p2_pos, combined_intensity):
    """Check if conditions indicate active rally"""
    if p1_pos is None or p2_pos is None:
        return False

    # Intensity should be somewhat high
    ACTIVE_INTENSITY_THRESHOLD = 0.030
    if combined_intensity < ACTIVE_INTENSITY_THRESHOLD:
        return False

    # Players should be closer to each other than in start position
    # and not in the static start positions
    if not is_rally_start(p1_pos, p2_pos, 0.0, 0.0):  # Not in start positions
        return True

    return False


def is_rally_end(combined_intensity, p1_pos, p2_pos):
    """Check if conditions indicate rally end"""
    # Intensity decreases
    END_INTENSITY_THRESHOLD = 0.010
    if combined_intensity > END_INTENSITY_THRESHOLD:
        return False

    if p1_pos[0] < 3.2 and p2_pos[0] > 3.2:
        return True

    # Or simply low intensity for extended period
    return combined_intensity < END_INTENSITY_THRESHOLD


def update_rally_state(
    current_state, combined_intensity, p1_pos, p2_pos, p1_intensity, p2_intensity
):
    """Update rally state based on new conditions and logical transitions"""

    new_state = current_state
    state_changed = False

    # Logical state transitions only:
    # rally_end -> rally_start -> rally_active -> rally_end

    if current_state == "rally_end":
        # Can only transition to rally_start
        if is_rally_start(p1_pos, p2_pos, p1_intensity, p2_intensity):
            # Must maintain start conditions for at least interval frames
            # if state_duration >= interval:
            new_state = "rally_start"
            state_changed = True

    elif current_state == "rally_start":
        # Can only transition to rally_active
        if is_rally_active(p1_pos, p2_pos, combined_intensity):
            new_state = "rally_active"
            state_changed = True

    elif current_state == "rally_active":
        # Can only transition to rally_end
        if is_rally_end(combined_intensity, p1_pos, p2_pos):
            new_state = "rally_end"
            state_changed = True

    return new_state, state_changed


def process_video(video_path, config_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Initialize court calibrator
    court_calibrator = CourtCalibrator(config["court_calibrator"])

    # Get first frame for homography computation
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame from video")

    # Compute homography matrix from first frame
    try:
        homography = court_calibrator.compute_homography(first_frame)
        print("Homography matrix computed successfully")
    except Exception as e:
        print(f"Error computing homography: {e}")
        return

    # Initialize player tracker with homography
    player_tracker = PlayerTracker(homography, config["player_tracker"])

    # Get processing parameters
    interval = config["processing"]["average_position_interval"]

    # Variables for tracking average positions
    frame_count = 0
    player1_positions = []
    player2_positions = []
    current_p1_avg = None
    current_p2_avg = None

    # Variables for tracking movement intensity
    player1_movements = []
    player2_movements = []
    player1_last_pos = None
    player2_last_pos = None
    current_p1_intensity = None
    current_p2_intensity = None

    # Rally segmentation variables
    rally_state = "rally_end"  # Start assuming rally has ended
    rally_state_duration = 0
    rally_transitions = []
    previous_player_distance = float("inf")
    current_player_distance = float("inf")

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Track players in current frame
        pixel_coords, real_coords = player_tracker.process_frame(frame)

        # Store real-world positions for averaging and calculate movement
        if 1 in real_coords and len(real_coords[1]) > 0:
            current_pos = real_coords[1][-1]
            player1_positions.append(current_pos)

            # Calculate movement intensity (distance moved since last frame)
            if player1_last_pos is not None:
                distance = np.sqrt(
                    (current_pos[0] - player1_last_pos[0]) ** 2
                    + (current_pos[1] - player1_last_pos[1]) ** 2
                )
                player1_movements.append(distance)

            player1_last_pos = current_pos

        if 2 in real_coords and len(real_coords[2]) > 0:
            current_pos = real_coords[2][-1]
            player2_positions.append(current_pos)

            # Calculate movement intensity (distance moved since last frame)
            if player2_last_pos is not None:
                distance = np.sqrt(
                    (current_pos[0] - player2_last_pos[0]) ** 2
                    + (current_pos[1] - player2_last_pos[1]) ** 2
                )
                player2_movements.append(distance)

            player2_last_pos = current_pos

        # Calculate combined intensity for display
        combined_intensity = 0.0
        if current_p1_intensity is not None and current_p2_intensity is not None:
            combined_intensity = (current_p1_intensity + current_p2_intensity) / 2
        elif current_p1_intensity is not None:
            combined_intensity = current_p1_intensity
        elif current_p2_intensity is not None:
            combined_intensity = current_p2_intensity

        # Every interval frames, update average positions and rally state
        if frame_count % interval == 0:
            # Calculate and print Player 1 average position and movement intensity
            if player1_positions:
                # Get last interval positions (or all if less than interval)
                recent_p1_positions = player1_positions[-interval:]
                avg_x1 = np.mean([pos[0] for pos in recent_p1_positions])
                avg_y1 = np.mean([pos[1] for pos in recent_p1_positions])
                current_p1_avg = (avg_x1, avg_y1)

                # Calculate movement intensity
                if player1_movements:
                    recent_p1_movements = player1_movements[
                        -(interval - 1) :
                    ]  # interval-1 because movements = positions-1
                    avg_intensity1 = np.mean(recent_p1_movements)
                    current_p1_intensity = avg_intensity1
                    print(
                        f"Player 1 - Avg pos: ({avg_x1:.2f}, {avg_y1:.2f})m, Avg intensity: {avg_intensity1:.3f}m/frame"
                    )
                else:
                    current_p1_intensity = 0.0
                    print(
                        f"Player 1 - Avg pos: ({avg_x1:.2f}, {avg_y1:.2f})m, Avg intensity: 0.000m/frame"
                    )
            else:
                current_p1_avg = None
                current_p1_intensity = None
                print("Player 1: No positions detected in recent frames")

            # Calculate and print Player 2 average position and movement intensity
            if player2_positions:
                # Get last interval positions (or all if less than interval)
                recent_p2_positions = player2_positions[-interval:]
                avg_x2 = np.mean([pos[0] for pos in recent_p2_positions])
                avg_y2 = np.mean([pos[1] for pos in recent_p2_positions])
                current_p2_avg = (avg_x2, avg_y2)

                # Calculate movement intensity
                if player2_movements:
                    recent_p2_movements = player2_movements[
                        -(interval - 1) :
                    ]  # interval-1 because movements = positions-1
                    avg_intensity2 = np.mean(recent_p2_movements)
                    current_p2_intensity = avg_intensity2
                    print(
                        f"Player 2 - Avg pos: ({avg_x2:.2f}, {avg_y2:.2f})m, Avg intensity: {avg_intensity2:.3f}m/frame"
                    )
                else:
                    current_p2_intensity = 0.0
                    print(
                        f"Player 2 - Avg pos: ({avg_x2:.2f}, {avg_y2:.2f})m, Avg intensity: 0.000m/frame"
                    )
            else:
                current_p2_avg = None
                current_p2_intensity = None
                print("Player 2: No positions detected in recent frames")
            # Update rally state (only check every interval frames)
            new_rally_state, state_changed = update_rally_state(
                rally_state,
                combined_intensity,
                current_p1_avg,
                current_p2_avg,
                current_p1_intensity,
                current_p2_intensity,
            )

            # Handle state transitions
            if state_changed:
                rally_transitions.append(
                    {
                        "frame": frame_count,
                        "from_state": rally_state,
                        "to_state": new_rally_state,
                        "combined_intensity": combined_intensity,
                        "player_distance": current_player_distance,
                    }
                )
                print(
                    f"\n*** RALLY STATE CHANGE at frame {frame_count}: {rally_state} -> {new_rally_state} ***"
                )
                rally_state = new_rally_state
                rally_state_duration = 0
            else:
                rally_state_duration += interval  # Increment by interval since we check every interval frames

            print(f"\n--- Frame {frame_count} ---")
            print(
                f"Rally State: {rally_state.upper()} (duration: {rally_state_duration} frames)"
            )
            print(f"Combined Intensity: {combined_intensity:.3f} m/frame")
            if current_player_distance != float("inf"):
                print(f"Player Distance: {current_player_distance:.2f} m")

        # Draw information on frame
        display_frame = frame.copy()

        # Draw bounding boxes and player IDs
        for player_id, bbox in pixel_coords.items():
            x1, y1, x2, y2 = map(int, bbox)

            # Choose color based on player ID
            color = (
                (0, 255, 0) if player_id == 1 else (255, 0, 0)
            )  # Green for P1, Blue for P2

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Draw player ID label
            label = f"Player {player_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Draw background rectangle for label
            cv2.rectangle(
                display_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                display_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Draw frame number
        cv2.putText(
            display_frame,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Draw rally state with color coding
        state_colors = {
            "rally_end": (0, 0, 255),  # Red
            "rally_start": (0, 255, 255),  # Yellow
            "rally_active": (0, 255, 0),  # Green
        }
        state_color = state_colors.get(rally_state, (255, 255, 255))
        rally_text = f"Rally State: {rally_state.upper()}"
        cv2.putText(
            display_frame,
            rally_text,
            (display_frame.shape[1] - 400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            state_color,
            2,
        )

        # Draw combined intensity
        intensity_text = f"Combined Intensity: {combined_intensity:.3f}"
        cv2.putText(
            display_frame,
            intensity_text,
            (display_frame.shape[1] - 400, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw average positions and movement intensity if available
        y_offset = 70
        if current_p1_avg is not None:
            p1_pos_text = f"P1 Avg: ({current_p1_avg[0]:.2f}, {current_p1_avg[1]:.2f})m"
            cv2.putText(
                display_frame,
                p1_pos_text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y_offset += 25

            if current_p1_intensity is not None:
                p1_intensity_text = f"P1 Intensity: {current_p1_intensity:.3f}m/frame"
                cv2.putText(
                    display_frame,
                    p1_intensity_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            y_offset += 30
        else:
            cv2.putText(
                display_frame,
                "Player 1: No data",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            y_offset += 55

        if current_p2_avg is not None:
            p2_pos_text = f"P2 Avg: ({current_p2_avg[0]:.2f}, {current_p2_avg[1]:.2f})m"
            cv2.putText(
                display_frame,
                p2_pos_text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y_offset += 25

            if current_p2_intensity is not None:
                p2_intensity_text = f"P2 Intensity: {current_p2_intensity:.3f}m/frame"
                cv2.putText(
                    display_frame,
                    p2_intensity_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        else:
            cv2.putText(
                display_frame,
                "Player 2: No data",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Display the frame
        cv2.imshow("Squash Player Tracking", display_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print rally segmentation summary
    print(f"\nProcessing completed. Total frames processed: {frame_count}")
    print(f"Final rally state: {rally_state}")
    print(f"\n=== RALLY SEGMENTATION SUMMARY ===")
    print(f"Total state transitions: {len(rally_transitions)}")

    if rally_transitions:
        print("\nState transition timeline:")
        for i, transition in enumerate(rally_transitions):
            frame = transition["frame"]
            from_state = transition["from_state"]
            to_state = transition["to_state"]
            intensity = transition["combined_intensity"]
            timestamp = frame / 30.0  # Assuming 30fps
            print(
                f"{i+1:2d}. Frame {frame:4d} ({timestamp:6.2f}s): {from_state:11s} -> {to_state:11s} (intensity: {intensity:.3f})"
            )
    else:
        print("No rally state transitions detected.")


if __name__ == "__main__":
    # Example usage
    video_path = "input.mp4"  # Replace with actual video path
    config_path = "config.json"

    try:
        process_video(video_path, config_path)
    except Exception as e:
        print(f"Error processing video: {e}")
