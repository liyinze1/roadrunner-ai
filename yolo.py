import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load the model
interpreter = Interpreter(model_path='models/yolo11n_float16.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

interpreter = Interpreter(model_path='models/yolo11n_float32.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)