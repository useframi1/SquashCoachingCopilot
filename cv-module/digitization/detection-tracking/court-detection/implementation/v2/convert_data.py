import os
from pathlib import Path


def convert_labels(input_dir, output_dir):
    """
    Convert Roboflow pose labels to YOLOv8-pose format.

    Input format (11 cols): class x_c y_c kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y
    Output format (17 cols): class x_c y_c w h kp1_x kp1_y v1 kp2_x kp2_y v2 kp3_x kp3_y v3 kp4_x kp4_y v4
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for label_file in input_path.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        converted_lines = []
        for line in lines:
            parts = line.strip().split()

            if len(parts) != 11:
                print(
                    f"Warning: {label_file.name} has {len(parts)} columns, expected 11"
                )
                continue

            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])

            # Extract keypoints
            keypoints = []
            for i in range(4):
                kp_x = float(parts[3 + i * 2])
                kp_y = float(parts[4 + i * 2])
                keypoints.extend([kp_x, kp_y])

            # Calculate bounding box from keypoints
            kp_xs = [float(parts[3 + i * 2]) for i in range(4)]
            kp_ys = [float(parts[4 + i * 2]) for i in range(4)]

            x_min = min(kp_xs)
            x_max = max(kp_xs)
            y_min = min(kp_ys)
            y_max = max(kp_ys)

            width = x_max - x_min
            height = y_max - y_min

            # Add some padding (optional, 10%)
            padding = 0.1
            width *= 1 + padding
            height *= 1 + padding

            # Build new line with visibility flags (2 = visible)
            new_parts = [
                class_id,
                str(x_center),
                str(y_center),
                str(width),
                str(height),
            ]
            for i in range(4):
                kp_x = parts[3 + i * 2]
                kp_y = parts[4 + i * 2]
                visibility = "2"  # Assume all keypoints are visible
                new_parts.extend([kp_x, kp_y, visibility])

            converted_lines.append(" ".join(new_parts))

        # Write converted labels
        output_file = output_path / label_file.name
        with open(output_file, "w") as f:
            f.write("\n".join(converted_lines) + "\n")

    print(f"✅ Converted {len(list(input_path.glob('*.txt')))} label files")


# Convert train and test labels
convert_labels("train/labels", "train/labels")

convert_labels("test/labels", "test/labels")

# Backup original labels and replace
# import shutil

# for split in ["train", "test"]:
#     labels_dir = f"/home/g03-s2025/Downloads/Full_Court/{split}/labels"
#     backup_dir = f"/home/g03-s2025/Downloads/Full_Court/{split}/labels_original"
#     converted_dir = f"/home/g03-s2025/Downloads/Full_Court/{split}/labels_converted"

#     # Backup original
#     if os.path.exists(labels_dir):
#         shutil.move(labels_dir, backup_dir)

#     # Use converted labels
#     shutil.move(converted_dir, labels_dir)

# print("✅ Labels converted and replaced!")
