import os

# Path to your test labels folder
labels_dir = r"C:\Users\hp\OneDrive\Desktop\Uni\Thesis\SquashCoachingCopilot\cv-module\digitization\detection-tracking\court-detection\implementation\v1\dataset_2\test\labels"

# Loop over all label files
for filename in os.listdir(labels_dir):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(labels_dir, filename)
    new_lines = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            if class_id == 0:
                # Keep the label and optionally rename class ID to 0 (or any new ID)
                parts[0] = "0"  # change to "0", "floor", or any number you prefer
                new_lines.append(" ".join(parts))

    # Overwrite file with filtered labels
    with open(file_path, "w") as f:
        f.write("\n".join(new_lines))

print("Labels updated: only class 0 kept and renamed")
