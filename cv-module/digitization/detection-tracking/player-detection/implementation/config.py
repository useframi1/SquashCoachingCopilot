CONFIG = {
    "paths": {
        "test_video": "testing/video-3.mp4",
        "coco_annotations": "data/_annotations.coco.json",
        "output_video": "testing/tracking_output_2.mp4",
        "output_results": "testing/tracking_results.txt",
    },
    "models": {
        "yolo_model": "models/yolov8m.pt",
        "reid_feature_size": 512,
        "reid_input_size": [224, 224],
    },
    "tracker": {
        "max_history": 30,
        "reid_threshold": 0.3,
        "reid_weight": 0.4,
        "position_weight": 0.6,
    },
    "evaluation": {
        "iou_threshold": 0.5,
        "player_1_class_id": 1,
        "player_2_class_id": 2,
    },
    "processing": {"max_frames": None, "progress_interval": 100},
    "visualization": {
        "display": True,
        "player_1_color": [0, 255, 0],
        "player_2_color": [0, 0, 255],
        "circle_radius": 5,
        "bbox_thickness": 2,
        "window_name": "Tracking Evaluation",
    },
    "output": {"video_codec": "mp4v", "results_precision": 3},
    "frame_formatting": {
        "video_name": "clip2",
        "pattern": "{video_name}_mp4-{frame_number:04d}.jpg",
    },
}
