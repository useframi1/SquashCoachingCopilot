from ultralytics import YOLO

model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)

model.train(data="dataset/data.yaml", epochs=150, imgsz=640)  # train the model

model.val()  # evaluate model performance on the validation set
