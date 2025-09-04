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
import numpy as np
from PIL import Image

def count_mask_pixels_from_yolo_output(output, proto, image_size, conf_thresh=0.5, mask_thresh=0.5):
    """
    Count foreground pixels in YOLOv11n-seg masks using NumPy + PIL only.

    Args:
        output: ndarray of shape (1, 37, 8400) — YOLO model output
        proto: ndarray of shape (H_proto, W_proto, C) — prototype mask (without batch dim)
        image_size: (H_img, W_img) — target size of masks (e.g., original image size)
        conf_thresh: float — minimum confidence to keep detection
        mask_thresh: float — binarization threshold for the mask

    Returns:
        List[int] — number of foreground pixels in each valid mask
    """
    output = output[0]  # Remove batch dim → shape: (37, 8400)
    H_img, W_img = image_size

    # Detect mask channel count from output
    total_channels, num_anchors = output.shape  # total_channels = 37
    num_mask_coeffs = total_channels - 6  # 4 bbox + 1 obj + 1 cls = 6 → rest are mask coeffs

    # Parse model outputs
    objectness = output[4, :]                          # (8400,)
    class_conf = output[5, :]                          # (8400,)
    mask_coeffs_all = output[6:6+num_mask_coeffs, :].T # (8400, num_mask_coeffs)

    # Final object confidence
    conf = objectness * class_conf                     # (8400,)
    keep = conf > conf_thresh
    mask_coeffs = mask_coeffs_all[keep]

    if mask_coeffs.shape[0] == 0:
        return []

    # 🔧 Fix proto if it has more channels than mask_coeffs
    if proto.shape[2] > mask_coeffs.shape[1]:
        proto = proto[:, :, :mask_coeffs.shape[1]]  # Trim proto to match coeffs
    elif proto.shape[2] < mask_coeffs.shape[1]:
        raise ValueError('mask_coeffs has more channels than proto — cannot compute mask.')

    # Flatten proto for dot product
    proto_flat = proto.reshape(-1, mask_coeffs.shape[1])  # (H_proto * W_proto, C)

    # Mask generation: (N, C) x (C, H*W) → (N, H*W)
    masks = np.dot(mask_coeffs, proto_flat.T)
    masks = 1 / (1 + np.exp(-masks))                       # Sigmoid
    H_proto, W_proto = proto.shape[:2]
    masks = masks.reshape(-1, H_proto, W_proto)            # (N, H_proto, W_proto)

    # Resize and count pixels
    pixel_counts = []
    for i in range(masks.shape[0]):
        mask_np = masks[i]
        pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8))
        pil_resized = pil_mask.resize((W_img, H_img), resample=Image.BILINEAR)
        mask_resized_np = np.array(pil_resized) / 255.0
        binary_mask = (mask_resized_np > mask_thresh).astype(np.uint8)
        pixel_counts.append(int(binary_mask.sum()))

    return sum(pixel_counts) / (H_img * W_img)

def extract_binary_masks(output, proto, image_size, conf_thresh=0.25, mask_thresh=0.5):
    output = output[0]
    total_channels = output.shape[0]
    num_mask_coeffs = total_channels - 6  # 4 bbox + 1 obj + 1 cls

    objectness = output[4, :]
    class_conf = output[5, :]
    mask_coeffs_all = output[6:6+num_mask_coeffs, :].T

    conf = objectness * class_conf
    keep = conf > conf_thresh
    mask_coeffs = mask_coeffs_all[keep]

    if mask_coeffs.shape[0] == 0:
        return []

    # Trim proto channels to match coeffs if needed
    if proto.shape[2] > mask_coeffs.shape[1]:
        proto = proto[:, :, :mask_coeffs.shape[1]]
    elif proto.shape[2] < mask_coeffs.shape[1]:
        raise ValueError('mask_coeffs > proto channels')

    proto_flat = proto.reshape(-1, mask_coeffs.shape[1])
    masks = np.dot(mask_coeffs, proto_flat.T)
    masks = 1 / (1 + np.exp(-masks))  # sigmoid
    H_proto, W_proto = proto.shape[:2]
    masks = masks.reshape(-1, H_proto, W_proto)

    # Resize to image size and binarize
    H_img, W_img = image_size
    binary_masks = []
    for i in range(masks.shape[0]):
        pil_mask = Image.fromarray((masks[i] * 255).astype(np.uint8))
        pil_resized = pil_mask.resize((W_img, H_img), resample=Image.BILINEAR)
        mask_resized_np = np.array(pil_resized) / 255.0
        binary_mask = (mask_resized_np > mask_thresh).astype(np.uint8)
        binary_masks.append(binary_mask)

    return binary_masks  # list of (H, W) binary np.uint8 masks

def save_combined_mask(masks, output_path='combined_mask.png'):
    if not masks:
        print('No masks to save.')
        return

    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for i, mask in enumerate(masks):
        combined += mask * (i + 1)  # Assign each mask a unique label

    # Convert to color (optional)
    img = Image.fromarray(combined * 40)  # scale to make mask visible
    img.save(output_path)
    print(f'Saved combined mask to: {output_path}')


def main():
    input_data = preprocess_image('water.jpg', input_shape)

    # Run inference with timing
    start = time()
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])  # (1, 37, 8400)
    proto  = interpreter.get_tensor(output_details[1]['index']) 
    
    output = np.array(output)
    proto = np.array(proto)[0]  # remove batch → (160, 160, 32)

    pixel_counts = count_mask_pixels_from_yolo_output(output, proto, image_size=(640, 640))
    
    print(f'Output shape: {output.shape}')
    print(f'Proto shape: {proto.shape}')
    print(f'Pixel counts: {pixel_counts}')
    
    masks = extract_binary_masks(output, proto, image_size=(640, 640))
    save_combined_mask(masks, output_path='combined_mask.png')
    

    inference_time = time() - start
    print(f'Inference time: {inference_time:.3f} seconds')
    # for box, cls, conf in zip(boxes, class_ids, confidences):
    #     print(f'water: {conf:.2f}, box: {box}')

if __name__ == '__main__':
    main()
