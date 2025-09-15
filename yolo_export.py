from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO('models/best.pt')  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

model.export(format='tflite', imgsz=640)  # export the model to TensorFlow Lite format
