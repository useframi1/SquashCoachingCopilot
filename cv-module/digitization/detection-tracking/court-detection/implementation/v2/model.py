# yolov8_multiclass_detection.py
from ultralytics import YOLO


def main():
    # Load a pretrained YOLOv8 model (nano = fastest, you can switch to yolov8s.pt, m.pt, l.pt, etc.)
    model = YOLO("yolov8m-pose.pt")  # or your downloaded pose weights file

    # Train the model on your multi-class dataset
    model.train(
        data="data.yaml",  # Roboflow dataset config (must include both class names)
        epochs=100,
        imgsz=640,
        batch=16,
    )

    # Validate on test set (gets metrics for each class + overall)
    metrics = model.val()
    print("📊 Evaluation Metrics:")
    print(metrics)

    # Extra: show per-class results
    for k, v in metrics.results_dict.items():
        print(f"{k}: {v}")

    # Predict on test images
    results = model.predict("test/images", save=True)
    print("✅ Predictions saved in runs/detect/predict")


if __name__ == "__main__":
    main()
