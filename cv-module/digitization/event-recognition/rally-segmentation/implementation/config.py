# Configuration for Rally State Prediction Pipeline

CONFIG = {
    "window_size": 50,  # Frames per aggregation window
    "active_model": "ml_based",
    "annotations": {
        "data_path": "data/annotations",  # Path to CSV files with base metrics
        "video_path": "videos/video-3.mp4",  # Path to video for annotation
    },
    "analysis": {
        "eda_plot_path": "analysis/rally_segmentation_eda.png",
    },
    "feature_engineering": {
        "lookback_frames": 3,  # Number of previous frames for lag features
        "court_center_x": 3.2,
        "service_line_y": 5.44,
    },
    "modeling": {
        "ml_based": {
            "model_path": "models/rally_state_model.pkl",
            "model_type": "xgboost",
        },
        "rule_based": {
            "lookback_frames": 2,
            "start_state": {
                "distance_min": 3.0,  # meters
                "distance_max": 4.0,  # meters
            },
            "active_state": {
                "distance_max": 3.5,  # meters - below this threshold
            },
            "end_state": {
                "distance_min": 3.5,  # meters - above this threshold
            },
            "court_center_x": 3.2,
            "service_line_y": 5.44,
            "service_boxes": {
                "left": {"x_min": 0.0, "x_max": 1.6, "y_min": 5.44, "y_max": 7.04},
                "right": {"x_min": 4.8, "x_max": 6.4, "y_min": 5.44, "y_max": 7.04},
            },
        },
        "test_size": 0.2,  # Fraction of videos for testing
        "random_seed": 42,
    },
    "evaluator": {
        "video_filter": "video-2",
        "plot_output_path": "testing/evaluation_results/predictions_plot.png",
        "min_duration": {
            "start": 1,
            "active": 4,
            "end": 2,
        },
        "tolerance_frames": 2,
    },
    "inference": {
        "video_path": "videos/video-2.mp4",
        "start_frame": 0,
        "end_frame": None,
        "save_output": True,
        "video_output_path": "real-time/results/output.mp4",
        "predictions_output_path": "real-time/results/predictions.csv",
    },
    "player_tracker": {
        "model_path": "models/yolov8m.pt",
        "player_class_id": 0,
        "player_class_name": "player",
    },
    "court_calibrator": {
        "model_path": "models/court_detection.pt",
        "real_coords": [[1.6, 5.44], [4.8, 5.44], [4.8, 7.04], [1.6, 7.04]],
        "confidence_threshold": 0.5,
    },
}
