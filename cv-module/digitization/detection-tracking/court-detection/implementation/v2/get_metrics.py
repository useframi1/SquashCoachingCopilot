from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/pose/court-detection/weights/best.pt")

# Validate on your dataset
results = model.val(data="dataset/data.yaml")

with open("metrics.txt", "w") as f:
    f.write("--- Overall Pose Metrics ---\n")
    f.write(f"Precision: {results.pose.mean_results()[0]:.4f}\n")
    f.write(f"Recall:    {results.pose.mean_results()[1]:.4f}\n")
    f.write(f"mAP@0.5:   {results.pose.mean_results()[2]:.4f}\n")
    f.write(f"mAP@0.5-0.95: {results.pose.mean_results()[3]:.4f}\n\n")

    f.write("--- Per-Class Pose Metrics ---\n")
    for i, name in results.names.items():
        p, r, map50, map5095 = results.pose.class_result(i)
        f.write(f"{name:10s} | P: {p:.4f} | R: {r:.4f} | mAP50: {map50:.4f} | mAP50-95: {map5095:.4f}\n")

print("Pose metrics saved to metrics.txt ✅")
