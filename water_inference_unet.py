import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
# If using full TensorFlow:
# from tensorflow.lite import Interpreter

# ---------------------------------------------------------
# User parameters
# ---------------------------------------------------------
MODEL_PATH = 'models/tiny_unet_int8.tflite'
IMAGE_PATH = 'water_720.png'      # any size, any format
MASK_THRESHOLD = 0.5          # segmentation threshold
# ---------------------------------------------------------

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
print('input details', input_details)
output_details = interpreter.get_output_details()
print('output details', output_details)

# Extract quantization info
input_scale, input_zero = input_details[0]['quantization']
output_scale, output_zero = output_details[0]['quantization']

# Model expects NHWC = (1, 256, 256, 3)
_, input_h, input_w, input_c = input_details[0]['shape']

print('Model input:', input_h, input_w, input_c)
print('Input quant:', input_scale, input_zero)
print('Output quant:', output_scale, output_zero)

# ---------------------------------------------------------
# Load + normalize + resize image automatically
# ---------------------------------------------------------
img = Image.open(IMAGE_PATH)

# Ensure RGB format
if img.mode != 'RGB':
    img = img.convert('RGB')

# Resize to model input size
img = img.resize((input_w, input_h), Image.BILINEAR)

# Convert to numpy
img_np = np.array(img, dtype=np.float32)

# ---------------------------------------------------------
# Quantize image to int8
# q = f/scale + zero_point
# ---------------------------------------------------------
img_q = img_np / 255.0               # normalize [0,1]
img_q = img_q / input_scale + input_zero
img_q = np.clip(img_q, -128, 127).astype(np.int8)

# Add batch dimension
input_tensor = np.expand_dims(img_q, axis=0)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_tensor)

# Run inference
interpreter.invoke()

# ---------------------------------------------------------
# Retrieve output
# ---------------------------------------------------------
output_q = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: 256x256x1

# Dequantize: f = (q - zero_point) * scale
output_f = (output_q.astype(np.float32) - output_zero) * output_scale
output_f = np.squeeze(output_f)   # shape: 256x256

# ---------------------------------------------------------
# Apply threshold to produce segmentation mask
# ---------------------------------------------------------
mask = (output_f > MASK_THRESHOLD).astype(np.uint8) * 255

mask_img = Image.fromarray(mask)
mask_img.save('segmentation_mask.png')

print('Segmentation mask saved as segmentation_mask.png')
