from pipeline import Pipeline

# Initialize the integrated pipeline
pipeline = Pipeline()

# Process video with all four pipelines
pipeline.process_video(
    video_path="video-5.mp4",
    output_path="output.mp4",  # Optional: save processed video
    display=True  # Display video in real-time
)
