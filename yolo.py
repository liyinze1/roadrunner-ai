import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# Load TFLite model
interpreter = Interpreter(model_path='yolo11n_float32.tflite')
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']

# Load and preprocess image
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((input_shape[2], input_shape[1]))
    image_np = np.array(image_resized, dtype=np.float32)
    image_np /= 255.0  # Normalize to 0.0–1.0
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np

# Example usage
image_path = 'bus.jpg'
input_data = preprocess_image(image_path, input_shape)

# Run inference
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()

# Get output
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])

print('Output shape:', output_data.shape)
