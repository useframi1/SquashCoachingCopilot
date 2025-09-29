# convert_to_yolo_pose.py
import os
from pathlib import Path


def convert_labels(input_dir, output_dir):
    """
    Convert keypoint-only labels to YOLOv8-pose format.

    Input (9 cols): class kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y
    Output (17 cols): class x_c y_c w h kp1_x kp1_y v1 kp2_x kp2_y v2 kp3_x kp3_y v3 kp4_x kp4_y v4
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    converted_count = 0

    for label_file in input_path.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        converted_lines = []
        for line in lines:
            parts = line.strip().split()

            if len(parts) != 9:
                print(
                    f"Warning: {label_file.name} has {len(parts)} columns, expected 9"
                )
                continue

            class_id = parts[0]

            # Extract all keypoints
            keypoints = []
            for i in range(4):
                kp_x = float(parts[1 + i * 2])
                kp_y = float(parts[2 + i * 2])
                keypoints.append((kp_x, kp_y))

            # Calculate bounding box from keypoints
            kp_xs = [kp[0] for kp in keypoints]
            kp_ys = [kp[1] for kp in keypoints]

            x_min = min(kp_xs)
            x_max = max(kp_xs)
            y_min = min(kp_ys)
            y_max = max(kp_ys)

            # Calculate center and size
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Add padding (10% on each side)
            padding = 0.1
            width *= 1 + padding
            height *= 1 + padding

            # Build YOLO-pose format: class x_c y_c w h kp1_x kp1_y v1 kp2_x kp2_y v2 ...
            new_parts = [
                class_id,
                f"{x_center:.10f}",
                f"{y_center:.10f}",
                f"{width:.10f}",
                f"{height:.10f}",
            ]

            # Add each keypoint with visibility flag (2 = visible)
            for kp_x, kp_y in keypoints:
                new_parts.extend(
                    [f"{kp_x:.10f}", f"{kp_y:.10f}", "2"]  # visibility: 2 = visible
                )

            converted_lines.append(" ".join(new_parts))

        # Write converted labels
        output_file = output_path / label_file.name
        with open(output_file, "w") as f:
            f.write("\n".join(converted_lines) + "\n")

        converted_count += 1

    print(f"✅ Converted {converted_count} label files")


# Main conversion
BASE_PATH = "dataset"

# Convert train labels
print("Converting train labels...")
convert_labels(f"{BASE_PATH}/train/labels", f"{BASE_PATH}/train/labels_converted")

# Convert test/val labels
print("Converting test labels...")
convert_labels(f"{BASE_PATH}/test/labels", f"{BASE_PATH}/test/labels_converted")

# # Backup and replace
# import shutil

# for split in ["train", "test"]:
#     labels_dir = f"{BASE_PATH}/{split}/labels"
#     backup_dir = f"{BASE_PATH}/{split}/labels_backup"
#     converted_dir = f"{BASE_PATH}/{split}/labels_converted"

#     # Backup original
#     if os.path.exists(labels_dir) and not os.path.exists(backup_dir):
#         print(f"Backing up original {split} labels...")
#         shutil.copytree(labels_dir, backup_dir)

#     # Remove old labels
#     if os.path.exists(labels_dir):
#         shutil.rmtree(labels_dir)

#     # Use converted labels
#     shutil.move(converted_dir, labels_dir)

print("\n✅ All labels converted!")
print("Original labels backed up to labels_backup folders")
print("\nNext steps:")
print("1. Delete cache files:")
print(f"   rm {BASE_PATH}/train/labels.cache")
print(f"   rm {BASE_PATH}/test/labels.cache")
print("2. Run your training script")
