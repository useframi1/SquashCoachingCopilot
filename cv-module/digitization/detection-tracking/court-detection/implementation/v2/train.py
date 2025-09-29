from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="court-detection",
    exist_ok=True,
)

results = model.val()
print("Validation results:", results)
