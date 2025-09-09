import os
import pandas as pd
import shutil
from PIL import Image

# Define dataset directories
dataset_dir = "yolo_dataset"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

train_images = os.path.join(train_dir, "images")
train_labels = os.path.join(train_dir, "labels")
test_images = os.path.join(test_dir, "images")
test_labels = os.path.join(test_dir, "labels")

# Ensure directories exist
for folder in [train_images, train_labels, test_images, test_labels]:
    os.makedirs(folder, exist_ok=True)

csv_path = "/Users/youssef/Documents/Thesis/Thesis-1/datasets-master/tboxes.csv"
images_dir = "/Users/youssef/Documents/Thesis/Thesis-1/datasets-master"

# Load CSV and remove nulls
df = pd.read_csv(csv_path, delimiter=";")
df = df.dropna()  # Remove rows with missing data

# Shuffle dataset for randomness
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]


def normalize(value, max_value):
    return value / max_value


def process_data(data, images_output, labels_output):
    processed_images, missing_images = 0, 0

    for index, row in data.iterrows():
        img_path = os.path.abspath(os.path.join(images_dir, row["filename"]))
        print(img_path)

        if not os.path.exists(img_path):
            print(f"Skipping {row['filename']}: File not found")
            missing_images += 1
            continue

        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Skipping {row['filename']}: Cannot open image ({e})")
            missing_images += 1
            continue

        # Raw keypoints (absolute pixel positions)
        keypoints_abs = [
            (row["tl.x"], row["tl.y"]),
            (row["tr.x"], row["tr.y"]),
            (row["br.x"], row["br.y"]),
            (row["bl.x"], row["bl.y"]),
        ]

        # Get bounding box from keypoints
        xs, ys = zip(*keypoints_abs)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Normalize bounding box
        center_x = normalize(bbox_center_x, img_width)
        center_y = normalize(bbox_center_y, img_height)
        width = normalize(bbox_width, img_width)
        height = normalize(bbox_height, img_height)

        # Start constructing label line
        kpt_line = [
            "0",
            f"{center_x:.6f}",
            f"{center_y:.6f}",
            f"{width:.6f}",
            f"{height:.6f}",
        ]

        # Normalize and add keypoints
        for x, y in keypoints_abs:
            x_norm = normalize(x, img_width)
            y_norm = normalize(y, img_height)
            v = 2
            kpt_line.extend([f"{x_norm:.6f}", f"{y_norm:.6f}", str(v)])

        label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(labels_output, label_filename)
        with open(label_path, "w") as f:
            f.write(" ".join(kpt_line))

        shutil.copy(img_path, os.path.join(images_output, os.path.basename(img_path)))
        processed_images += 1

    print(
        f"✅ Processed {processed_images} images, ⚠️ Skipped {missing_images} missing images"
    )


process_data(train_df, train_images, train_labels)
process_data(test_df, test_images, test_labels)

# Create data.yml file
data_yaml_content = f"""
train: {os.path.abspath(train_images)}
val: {os.path.abspath(test_images)}

nc: 1
kpt_shape: [4, 3]  # 4 keypoints, each with (x, y, v)
names: ["object"]
kp_names: ["top-left", "top-right", "bottom-right", "bottom-left"]
"""

with open(os.path.join(dataset_dir, "data.yml"), "w") as f:
    f.write(data_yaml_content.strip())

print("\n✅ YOLO keypoint dataset prepared successfully in:", dataset_dir)
