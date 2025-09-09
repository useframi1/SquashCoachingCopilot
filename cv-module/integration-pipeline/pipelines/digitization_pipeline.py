import streamlit as st
from utilities.court_calibrator import CourtCalibrator
from utilities.player_tracker import EnhancedPlayerTracker
from utilities.tracking_visualizer import TrackingVisualizer
from utilities.utils import load_config
import cv2
import time


class DigitizationPipeline:
    def __init__(
        self,
        video_path,
        court_dimensions,
        court_top_view_path,
        use_streamlit=False,
        streamlit_frame_placeholder=None,
    ):
        self.config = load_config("configs/digitization_pipeline_config.json")

        self.video_path = video_path
        self.court_dimensions = court_dimensions
        self.court_top_view_path = court_top_view_path
        self.use_streamlit = use_streamlit
        self.streamlit_frame_placeholder = streamlit_frame_placeholder

        self.calibrator = CourtCalibrator(config=self.config["court_calibrator"])
        self.tracker = None
        self.tracking_visualizer = TrackingVisualizer(
            court_dimensions=court_dimensions,
            court_top_view_path=court_top_view_path,
            config=self.config["tracking_visualizer"],
        )
        self.cap = None
        self.fps = None
        self.homography = None

    def initialize_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

    def calibrate_court(self):
        ret, first_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame from video.")
        self.homography = self.calibrator.compute_homography(first_frame)
        return first_frame

    def initialize_tracker(self, first_frame):
        self.tracker = EnhancedPlayerTracker(
            self.homography, config=self.config["player_tracker"]
        )
        self.tracker.process_frame(first_frame)

    def process_video_frames(self):
        max_frames = self.config["max_frames"]
        visualize = self.config["visualize"]
        cut_video = self.config["cut_video"]
        realtime = self.config["realtime"]
        frame_count = 1

        frame_time = 1.0 / self.fps if self.fps > 1 else 1.0 / 30

        while self.cap.isOpened():
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            pixel_positions, real_positions = self.tracker.process_frame(frame)

            if visualize:
                annotated = self.tracking_visualizer.visualize_tracking(
                    frame, pixel_positions, real_positions
                )

                if self.use_streamlit:
                    rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    if self.streamlit_frame_placeholder:
                        self.streamlit_frame_placeholder.image(
                            rgb_annotated, caption="Tracking", channels="RGB"
                        )
                    else:
                        st.image(rgb_annotated, caption="Tracking", channels="RGB")
                else:
                    cv2.imshow("Tracking", annotated)
                    if realtime:
                        elapsed = time.time() - start_time
                        wait_time = max(1, int((frame_time - elapsed) * 1000))
                        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                            print("Interrupted by user.")
                            break
                    else:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Interrupted by user.")
                            break

            frame_count += 1

    def finalize(self):
        self.cap.release()
        if not self.use_streamlit:
            cv2.destroyAllWindows()
        return self.tracker.get_all_positions(), self.fps

    def run(self):
        self.initialize_video()
        first_frame = self.calibrate_court()
        self.initialize_tracker(first_frame)
        self.process_video_frames()
        return self.finalize()
