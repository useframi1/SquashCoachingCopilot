import cv2
import json
import sys
from pathlib import Path

from court_calibrator import CourtCalibrator
from player_tracker import PlayerTracker
from rally_state_detector import RallyStateDetector
from player_analyzer import PlayerAnalyzer
from video_display import VideoDisplay
from rally_plotter import RallyPlotter


def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def initialize_components(config):
    """Initialize all system components"""
    try:
        # Initialize court calibrator
        court_calibrator = CourtCalibrator(config["court_calibrator"])

        # Initialize other components (will be created after homography)
        return {
            "court_calibrator": court_calibrator,
            "rally_detector": RallyStateDetector(config),
            "player_analyzer": PlayerAnalyzer(config),
            "video_display": VideoDisplay(config),
            "rally_plotter": RallyPlotter(config),
        }
    except KeyError as e:
        print(f"Error: Missing configuration key: {e}")
        sys.exit(1)


def compute_homography(court_calibrator, first_frame):
    """Compute homography matrix from the first frame"""
    try:
        homography = court_calibrator.compute_homography(first_frame)
        print("✓ Homography matrix computed successfully")
        return homography
    except Exception as e:
        print(f"✗ Error computing homography: {e}")
        return None


def print_rally_summary(rally_detector, total_frames, fps):
    """Print comprehensive rally segmentation summary"""
    summary = rally_detector.get_summary()

    print(f"\n{'='*50}")
    print("RALLY SEGMENTATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total frames processed: {total_frames}")
    print(f"Final rally state: {summary['final_state']}")
    print(f"Total state transitions: {summary['total_transitions']}")

    if summary["transitions"]:
        print(f"\nState transition timeline:")
        print(
            f"{'#':>2} {'Frame':>6} {'Time(s)':>8} {'From':>12} {'To':>12} {'Intensity':>10} {'Distance':>10}"
        )
        print("-" * 70)

        for i, transition in enumerate(summary["transitions"]):
            frame = transition["frame"]
            from_state = transition["from_state"]
            to_state = transition["to_state"]
            intensity = transition["combined_intensity"]
            distance = transition.get("player_distance", 0)
            timestamp = frame / fps

            print(
                f"{i+1:2d} {frame:6d} {timestamp:8.2f} {from_state:>12} {to_state:>12} "
                f"{intensity:10.3f} {distance:10.2f}"
            )
    else:
        print("No rally state transitions detected.")


def process_video(video_path, config_path):
    """Main video processing function"""
    print(f"🎾 Starting Squash Rally Segmentation")
    print(f"📹 Video: {video_path}")
    print(f"⚙️  Config: {config_path}")

    # Load configuration
    config = load_config(config_path)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Error: Could not open video file '{video_path}'")
        return False

    print(f"✓ Video opened successfully")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📊 Video info: {fps:.1f} FPS, {total_frames_in_video} total frames")

    # Initialize components
    components = initialize_components(config)

    # Get first frame for homography computation
    ret, first_frame = cap.read()
    if not ret:
        print("✗ Error: Could not read first frame from video")
        cap.release()
        return False

    # Compute homography
    homography = compute_homography(components["court_calibrator"], first_frame)
    if homography is None:
        cap.release()
        return False

    # Initialize player tracker with homography
    player_tracker = PlayerTracker(homography, config["player_tracker"])

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Processing variables
    frame_count = 0
    processing_interval = config["processing"]["average_position_interval"]

    print(f"🚀 Starting video processing (interval: {processing_interval} frames)")
    print("Press 'q' to quit, 'p' to pause/resume")

    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Track players in current frame
                pixel_coords, real_coords = player_tracker.process_frame(frame)

                # Update player analysis
                components["player_analyzer"].update_positions(real_coords)

                # Calculate averages every interval frames
                should_update = components["player_analyzer"].calculate_averages(
                    frame_count
                )

                if should_update:
                    # Print statistics
                    components["player_analyzer"].print_stats(frame_count)

                    # Update rally state
                    player_stats = components["player_analyzer"].get_player_stats()
                    combined_intensity = player_stats["combined_intensity"]

                    new_state, state_changed = components[
                        "rally_detector"
                    ].update_state(
                        combined_intensity,
                        player_stats["player1"]["avg_position"],
                        player_stats["player2"]["avg_position"],
                        frame_count,
                        player_stats["avg_distance"],
                    )

                    # Add data point to plotter
                    components["rally_plotter"].add_data_point(
                        frame_count,
                        player_stats["avg_distance"],
                        components["rally_detector"].current_state,
                        combined_intensity,
                    )

                    if state_changed:
                        print(
                            f"\n🔄 RALLY STATE CHANGE at frame {frame_count}: "
                            f"{components['rally_detector'].current_state}"
                        )

                        # Add transition to plotter
                        if components["rally_detector"].transitions:
                            latest_transition = components[
                                "rally_detector"
                            ].transitions[-1]
                            components["rally_plotter"].add_state_transition(
                                latest_transition
                            )

            # Create display frame
            player_stats = components["player_analyzer"].get_player_stats()
            combined_intensity = player_stats["combined_intensity"]

            display_frame = components["video_display"].create_display_frame(
                frame,
                frame_count,
                pixel_coords,
                components["rally_detector"].current_state,
                combined_intensity,
                player_stats,
            )

            # Show frame and handle user input
            key = components["video_display"].show_frame(display_frame)

            if key == ord("q"):
                print("\n👋 User requested quit")
                break
            elif key == ord("p"):
                paused = not paused
                print(f"⏸️  Video {'paused' if paused else 'resumed'}")
            elif key == ord(" "):  # Spacebar
                paused = not paused
                print(f"⏸️  Video {'paused' if paused else 'resumed'}")

    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        return False

    finally:
        # Cleanup
        cap.release()
        components["video_display"].cleanup()

        # Print summary
        print_rally_summary(components["rally_detector"], frame_count, fps)

        # Generate plots and analysis
        components["rally_plotter"].create_all_plots(fps)

        print(f"\n✅ Processing completed successfully!")

        return True


def main():
    """Main entry point"""
    # Default paths
    video_path = "input.mp4"
    config_path = "config.json"

    # Handle command line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        config_path = sys.argv[2]

    # Check if files exist
    if not Path(video_path).exists():
        print(f"❌ Error: Video file '{video_path}' not found")
        print(f"Usage: python {sys.argv[0]} [video_path] [config_path]")
        sys.exit(1)

    if not Path(config_path).exists():
        print(f"❌ Error: Config file '{config_path}' not found")
        sys.exit(1)

    # Process video
    success = process_video(video_path, config_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
