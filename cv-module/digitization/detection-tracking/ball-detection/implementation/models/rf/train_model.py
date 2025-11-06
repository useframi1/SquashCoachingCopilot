from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

model.train(data="dataset/data.yaml", epochs=100, imgsz=640)  # train the model

model.val()  # evaluate model performance on the validation set
