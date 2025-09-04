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

def count_mask_pixels_from_yolo_output(output, proto, image_size, conf_thresh=0.25, mask_thresh=0.5):
    """
    Count foreground pixels in masks from YOLOv11n-seg model output using only NumPy and PIL.

    Args:
        output: ndarray of shape (1, 37, 8400) — model output
        proto: ndarray of shape (160, 160, 32) — prototype masks (without batch dim)
        image_size: (H, W) — size of the input image
        conf_thresh: float — confidence threshold
        mask_thresh: float — binarization threshold for masks

    Returns:
        List of pixel counts per mask (int)
    """
    output = output[0]  # shape: (37, 8400)

    # Parse the outputs
    objectness = output[4, :]              # (8400,)
    class_conf = output[5, :]              # (8400,)
    mask_coeffs_all = output[6:, :].T      # (8400, 31)

    # Confidence score
    conf = objectness * class_conf         # (8400,)
    keep = conf > conf_thresh

    mask_coeffs = mask_coeffs_all[keep]
    if mask_coeffs.shape[0] == 0:
        return []

    # Proto: (H, W, C) → (H*W, C)
    proto_flat = proto.reshape(-1, proto.shape[2])  # (160*160, 32)

    # Dot product: (N, 31) x (31, H*W) → (N, H*W)
    masks = np.dot(mask_coeffs, proto_flat.T)       # (N, 160*160)
    masks = 1 / (1 + np.exp(-masks))                # sigmoid
    masks = masks.reshape(-1, proto.shape[0], proto.shape[1])  # (N, 160, 160)

    H, W = image_size
    pixel_counts = []
    for i in range(masks.shape[0]):
        # Resize with PIL
        mask_np = masks[i]
        pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8))  # convert to grayscale image
        pil_resized = pil_mask.resize((W, H), resample=Image.BILINEAR)
        mask_resized_np = np.array(pil_resized) / 255.0
        binary_mask = (mask_resized_np > mask_thresh).astype(np.uint8)
        pixel_counts.append(int(binary_mask.sum()))

    return pixel_counts

def main():
    input_data = preprocess_image('water.jpg', input_shape)

    # Run inference with timing
    start = time()
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    inference_time = time() - start

    # Get and process output
    output = np.array(output)
    proto = np.array(proto)[0]  # remove batch → (160, 160, 32)

    pixel_counts = count_mask_pixels_from_yolo_output(output, proto, image_size=(640, 640))
    
    print(f'Output shape: {output.shape}')
    print(f'Proto shape: {proto.shape}')
    print(f'Pixel counts: {pixel_counts}')
    

    print(f'Inference time: {inference_time:.3f} seconds')
    # for box, cls, conf in zip(boxes, class_ids, confidences):
    #     print(f'water: {conf:.2f}, box: {box}')

if __name__ == '__main__':
    main()
