from roboflow import Roboflow

rf = Roboflow(api_key="1IccxKnzEELTy6IxEwwC")
project = rf.workspace("projekt-squash-offac").project("squash-ball-detection-1lbti")
version = project.version(1)
dataset = version.download("yolov11", location="./dataset")
