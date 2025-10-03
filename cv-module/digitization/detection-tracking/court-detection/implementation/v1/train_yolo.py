from ultralytics import YOLO

# Path to the data YAML file
data_yaml = "yolo_dataset/data.yml"

# Path to the pre-trained model (e.g., yolov8n.pt or any other YOLOv8 model)
pretrained_model = "yolov8n.pt"

# Hyperparameters for training
epochs = 100
img_size = 640

# Create a YOLO object
model = YOLO(pretrained_model)  # Load a pre-trained YOLO model

# Start training
model.train(
    data=data_yaml,  # Path to the data YAML file
    epochs=epochs,  # Number of epochs to train
    imgsz=img_size,  # Image size
    project="runs/train",  # Directory to save training results
    name="keypoint_model",  # Name of the experiment
    exist_ok=True,  # Allow overwriting of existing experiment directory
)

# After training completes, the model will be saved in the specified "runs/train" folder
results = model.val()  # Validate the model on the validation set
print("Validation results:", results)  # Print validation results
