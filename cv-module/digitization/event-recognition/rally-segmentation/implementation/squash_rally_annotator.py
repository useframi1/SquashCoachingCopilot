import os
import cv2
import json
from datetime import timedelta
import time


class SquashAnnotator:
    def __init__(self, video_dir, output_file, sequence_length=64):
        # Configuration
        self.video_dir = video_dir
        self.output_file = output_file
        self.sequence_length = sequence_length
        self.half_seq_length = (
            sequence_length // 2
        )  # Half before, half after the marked frame

        # Load existing annotations if available
        self.annotations = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    self.annotations = json.load(f)
                print(f"Loaded existing annotations from {output_file}")
            except Exception as e:
                print(f"Error loading annotations: {e}")
                self.annotations = {}

        # Find video files
        self.video_files = self.find_video_files()
        print(f"Found {len(self.video_files)} video files")

    def find_video_files(self):
        """Find all video files in the directory"""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        video_files = []

        for file in os.listdir(self.video_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(self.video_dir, file)
                video_files.append(video_path)

        return video_files

    def annotate_videos(self):
        """Process each video for annotation"""
        for video_path in self.video_files:
            filename = os.path.basename(video_path)
            print(f"\nProcessing: {filename}")

            # Check if we already have annotations for this video
            if filename in self.annotations:
                choice = input(
                    f"Video already has {len(self.annotations[filename])} annotations. Continue annotating? (y/n): "
                )
                if choice.lower() != "y":
                    continue
            else:
                self.annotations[filename] = []

            # Start annotation process for this video
            self.annotate_video(video_path, filename)

            # Save after each video
            self.save_annotations()

    def annotate_video(self, video_path, filename):
        """Annotate a single video"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize window if video is too large
        window_name = "Video Annotation - Press 's' when a rally starts, 'e' when it ends, 'q' to quit, space to pause/play"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if width > 1280 or height > 720:
            cv2.resizeWindow(window_name, 1280, 720)

        # Annotation variables
        current_frame = 0
        playing = True
        play_speed = 1.0  # Normal speed

        print("\nControls:")
        print("  's' - Mark the start of a rally")
        print("  'e' - Mark the end of a rally")
        print("  'q' - Quit current video")
        print("  Space - Pause/Play")
        print("  '+' - Increase playback speed")
        print("  '-' - Decrease playback speed")
        print("  Left/Right Arrow - Step one frame back/forward (when paused)")

        # Main loop
        while True:
            # Set current frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break

            # Display frame information
            current_time = current_frame / fps
            time_str = str(timedelta(seconds=int(current_time))).split(".")[0]
            info_text = f"Frame: {current_frame}/{total_frames} | Time: {time_str} | Speed: {play_speed}x | Video: {filename}"

            # Create a copy of the frame to avoid modifying the original
            display_frame = frame.copy()

            # Display information bar
            cv2.rectangle(display_frame, (0, 0), (width, 30), (0, 0, 0), -1)
            cv2.putText(
                display_frame,
                info_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # Display the frame
            cv2.imshow(window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Quit on 'q'
            if key == ord("q"):
                break

            # Toggle play/pause on space
            elif key == 32:  # Space
                playing = not playing
                print("Playback:", "Playing" if playing else "Paused")

            # Mark rally start on 's'
            elif key == ord("s"):
                # Calculate start and end frames
                start_frame = max(0, current_frame - self.half_seq_length)
                end_frame = min(total_frames - 1, current_frame + self.half_seq_length)

                # Create annotation
                annotation = {
                    "label": "rally_start",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "center_frame": current_frame,  # For reference
                }

                # Add to annotations
                self.annotations[filename].append(annotation)

                print(
                    f"Marked rally start at frame {current_frame} (sequence: {start_frame}-{end_frame})"
                )

                # Visual indication of marking
                cv2.rectangle(display_frame, (0, 0), (width, height), (0, 255, 0), 10)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(500)  # Flash green for 500ms

            # Mark rally end on 'e'
            elif key == ord("e"):
                # Calculate start and end frames
                start_frame = max(0, current_frame - self.half_seq_length)
                end_frame = min(total_frames - 1, current_frame + self.half_seq_length)

                # Create annotation
                annotation = {
                    "label": "rally_end",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "center_frame": current_frame,  # For reference
                }

                # Add to annotations
                self.annotations[filename].append(annotation)

                print(
                    f"Marked rally end at frame {current_frame} (sequence: {start_frame}-{end_frame})"
                )

                # Visual indication of marking
                cv2.rectangle(display_frame, (0, 0), (width, height), (255, 0, 0), 10)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(500)  # Flash blue for 500ms

            # Add negative sample on 'n'
            elif key == ord("n"):
                # Calculate start and end frames
                start_frame = max(0, current_frame - self.half_seq_length)
                end_frame = min(total_frames - 1, current_frame + self.half_seq_length)

                # Create annotation
                annotation = {
                    "label": "in_play",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "center_frame": current_frame,  # For reference
                }

                # Add to annotations
                self.annotations[filename].append(annotation)

                print(
                    f"Marked NOT rally at frame {current_frame} (sequence: {start_frame}-{end_frame})"
                )

                # Visual indication of marking
                cv2.rectangle(display_frame, (0, 0), (width, height), (0, 0, 255), 10)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(500)  # Flash red for 500ms

            # Increase speed
            elif key == ord("+") or key == ord("="):
                play_speed = min(play_speed + 0.25, 4.0)
                print(f"Speed: {play_speed}x")

            # Decrease speed
            elif key == ord("-") or key == ord("_"):
                play_speed = max(play_speed - 0.25, 0.25)
                print(f"Speed: {play_speed}x")

            # Previous frame (when paused)
            elif key == 81 or key == 2:  # Left arrow
                if not playing:
                    current_frame = max(0, current_frame - 100)

            # Next frame (when paused)
            elif key == 83 or key == 3:  # Right arrow
                if not playing:
                    current_frame = min(total_frames - 1, current_frame + 100)

            # Update current frame if playing
            if playing:
                # Calculate delay based on speed
                delay = 1.0 / (fps * play_speed)
                time.sleep(
                    max(0.001, delay - 0.03)
                )  # Subtract processing time (approx)

                current_frame += 1

                # Check if we've reached the end
                if current_frame >= total_frames:
                    print("End of video reached")
                    break

        # Release resources
        cap.release()

        print(
            f"Completed annotation for {filename} - Added {len(self.annotations[filename])} annotations"
        )

    def save_annotations(self):
        """Save annotations to JSON file"""
        try:
            with open(self.output_file, "w") as f:
                json.dump(self.annotations, f, indent=2)

            total_annotations = sum(len(anns) for anns in self.annotations.values())
            print(f"Saved {total_annotations} annotations to {self.output_file}")
        except Exception as e:
            print(f"Error saving annotations: {e}")

    def run(self):
        """Main entry point"""
        print(
            f"Starting annotation with sequence length: {self.sequence_length} frames"
        )
        try:
            self.annotate_videos()
        except KeyboardInterrupt:
            print("\nAnnotation interrupted by user")
        finally:
            print("Saving final annotations...")
            self.save_annotations()
            cv2.destroyAllWindows()
            print("Done")
