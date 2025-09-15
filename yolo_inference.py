from ultralytics import YOLO

model = YOLO('models/water_float32.tflite', task='segment')

model.predict('water.jpg', save=True, imgsz=640, conf=0.5, name='river-predict')