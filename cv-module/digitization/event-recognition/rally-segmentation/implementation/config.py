# Configuration for Rally State Prediction Pipeline

CONFIG = {
    # Data paths
    "data_path": "data/annotations",  # Path to CSV files with base metrics
    "model_path": "models/rally_state_model.pkl",
    # Model configuration
    "model_type": "xgboost",  # Options: "xgboost", "random_forest"
    # Feature engineering
    "window_size": 50,  # Frames per aggregation window
    "lookback_frames": 3,  # Number of previous frames for lag features
    # Court configuration
    "court_center_x": 3.2,
    "service_line_y": 5.44,
    # Training
    "test_size": 0.2,  # Fraction of videos for testing
    "random_seed": 42,
    # Video processing (for inference)
    "video_fps": 30,
    "player_tracker_model": "models/yolov8m.pt",
    "court_detection_model": "models/court_detection.pt",
}
