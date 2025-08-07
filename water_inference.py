import numpy as np
from PIL import Image
from time import time
from tflite_runtime.interpreter import Interpreter

# Load TFLite model
interpreter = Interpreter(model_path='models/water_float32.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # (1, 640, 640, 3)

def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[2], input_shape[1]))
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def process_output(output_data, conf_threshold=0.25):
    output = output_data[0]  # (84, 8400)
    boxes = output[:4, :].T  # (8400, 4)
    objectness = output[4, :]
    class_probs = output[5:, :].T  # (8400, 80)
    scores = objectness[:, None] * class_probs  # (8400, 80)

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > conf_threshold
    return boxes[mask], class_ids[mask], confidences[mask]

def main():
    input_data = preprocess_image('water.jpg', input_shape)

    # Run inference with timing
    start = time()
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    inference_time = time() - start

    # Get and process output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f'Output shape: {output_data.shape}')
    
    # boxes, class_ids, confidences = process_output(output_data)

    print(f'Inference time: {inference_time:.3f} seconds')
    # for box, cls, conf in zip(boxes, class_ids, confidences):
    #     print(f'water: {conf:.2f}, box: {box}')

if __name__ == '__main__':
    main()
