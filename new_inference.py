import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import gc  # For garbage collection
import time
import os

class YOLOv11Segmentation:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.7, 
                 max_image_size=1024):
        """
        Initialize YOLOv11 segmentation model (supports both FLOAT32 and INT8)
        
        Args:
            model_path: Path to the .tflite model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_image_size = max_image_size
        
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors info
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        # Check if model is quantized
        self.input_dtype = self.input_details[0]['dtype']
        self.is_input_quantized = self.input_dtype in [np.uint8, np.int8]
        
        if self.is_input_quantized:
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
            print(f"✓ Quantized INPUT detected")
            print(f"  - Input dtype: {self.input_dtype}")
            print(f"  - Input scale: {self.input_scale}")
            print(f"  - Input zero point: {self.input_zero_point}")
        else:
            print(f"✓ Float INPUT detected (dtype: {self.input_dtype})")
        
        # Check output quantization
        self.output_quantization_info = []
        for i, output_detail in enumerate(self.output_details):
            output_dtype = output_detail['dtype']
            is_quantized = output_dtype in [np.uint8, np.int8]
            
            if is_quantized:
                scale, zero_point = output_detail['quantization']
                self.output_quantization_info.append({
                    'is_quantized': True,
                    'dtype': output_dtype,
                    'scale': scale,
                    'zero_point': zero_point
                })
                print(f"✓ Quantized OUTPUT {i} detected (dtype: {output_dtype}, scale: {scale}, zero_point: {zero_point})")
            else:
                self.output_quantization_info.append({
                    'is_quantized': False,
                    'dtype': output_dtype
                })
                print(f"✓ Float OUTPUT {i} detected (dtype: {output_dtype})")
        
        print(f"\nModel loaded successfully")
        print(f"Input shape: {self.input_shape}")
        
    def preprocess_image(self, image_path):
        """
        Memory-optimized image preprocessing (handles both FLOAT32 and INT8 models)
        """
        # Load and potentially resize image for memory efficiency
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image_path.convert('RGB')
            
        # Resize to model input size
        resized_image = original_image.resize((self.input_width, self.input_height), Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(resized_image, dtype=np.float32)
        del resized_image
        del original_image
        gc.collect()
        
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Apply quantization if model expects quantized input
        if self.is_input_quantized:
            # Quantize: float32 -> uint8/int8
            image_array = image_array / self.input_scale + self.input_zero_point
            image_array = np.clip(image_array, 0, 255).astype(self.input_dtype)
            print(f"Input quantized to {self.input_dtype}")
        
        return image_array
    
    def postprocess_detections(self, outputs):
        """
        Memory-optimized post-processing of YOLOv11 outputs (handles quantized outputs)
        """
        detections = []

        # Extract outputs with memory management
        detection_output = outputs[0][0]  # Could be (37, 8400) or (160, 160, 32)
        
        print(f"Raw detection output shape: {detection_output.shape}")
        print(f"Detection output dtype: {detection_output.dtype}, range: [{detection_output.min():.3f}, {detection_output.max():.3f}]")
        
        # Handle different output formats
        if len(detection_output.shape) == 3:
            # Format: (H, W, C) - need to reshape to (N, C)
            print(f"Detected 3D output format (H, W, C): {detection_output.shape}")
            H, W, C = detection_output.shape
            # Reshape to (H*W, C)
            detection_output = detection_output.reshape(-1, C)
            print(f"Reshaped to: {detection_output.shape}")
            
            # Check if we have mask prototypes
            if len(outputs) > 1:
                mask_protos = outputs[1][0]
                print(f"Mask prototypes shape: {mask_protos.shape}")
            else:
                mask_protos = None
                print("No mask prototypes found")
        else:
            # Format: (37, 8400) or (8400, 37)
            mask_protos = outputs[1][0] if len(outputs) > 1 else None
            
            # Transpose detection output if needed
            if detection_output.shape[0] < detection_output.shape[1]:
                print("Transposing detection output from (C, N) to (N, C)")
                detection_output = detection_output.T
            
            print(f"After transpose - Detection output shape: {detection_output.shape}")
        
        # Now detection_output should be (N, C) where N is number of detections
        num_detections, num_features = detection_output.shape
        print(f"Processing {num_detections} detections with {num_features} features each")
        
        # Determine the feature layout based on number of features
        if num_features == 32:
            # Format: [x, y, w, h, conf, mask_coeffs...]
            coord_indices = [0, 1, 2, 3]
            conf_index = 4
            mask_start_index = 5
            num_mask_coeffs = 27  # 32 - 5 = 27
        elif num_features >= 37:
            # Format: [x, y, w, h, conf, mask_coeffs...]
            coord_indices = [0, 1, 2, 3]
            conf_index = 4
            mask_start_index = 5
            num_mask_coeffs = min(32, num_features - 5)
        else:
            print(f"Warning: Unexpected number of features: {num_features}")
            coord_indices = [0, 1, 2, 3]
            conf_index = 4
            mask_start_index = 5
            num_mask_coeffs = max(0, num_features - 5)
        
        # Process detections in chunks to save memory
        chunk_size = min(8400, num_detections)
        
        for chunk_start in range(0, num_detections, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_detections)
            detection_chunk = detection_output[chunk_start:chunk_end]
            
            for i, detection in enumerate(detection_chunk):
                actual_i = chunk_start + i
                
                # Extract coordinates and confidence
                x_center = detection[coord_indices[0]]
                y_center = detection[coord_indices[1]]
                width = detection[coord_indices[2]]
                height = detection[coord_indices[3]]
                confidence = float(detection[conf_index])
                
                if confidence < self.conf_threshold:
                    continue
                
                # Convert coordinates efficiently
                x_center_px = x_center * 640
                y_center_px = y_center * 640
                width_px = width * 640
                height_px = height * 640
                
                x1 = (x_center_px - width_px/2)
                y1 = (y_center_px - height_px/2)
                x2 = (x_center_px + width_px/2)
                y2 = (y_center_px + height_px/2)
                
                # Bounds checking
                x1 = max(0, min(x1, 640))
                y1 = max(0, min(y1, 640))
                x2 = max(x1 + 1, min(x2, 640))
                y2 = max(y1 + 1, min(y2, 640))
                
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width > 10 and box_height > 10:
                    # Generate mask for all valid detections in low memory mode
                    mask = None
                    if mask_protos is not None and confidence > 0.5 and num_mask_coeffs > 0:
                        try:
                            mask_coeffs = detection[mask_start_index:mask_start_index + num_mask_coeffs]
                            mask = self.generate_mask_optimized(mask_coeffs, mask_protos, 
                                                              (x_center_px - width_px/2, y_center_px - height_px/2,
                                                               x_center_px + width_px/2, y_center_px + height_px/2))
                        except Exception as e:
                            print(f"Warning: Mask generation failed for detection {actual_i}: {e}")
                            mask = None
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': 0,
                        'mask': mask
                    })
            
            # Force garbage collection after each chunk
            gc.collect()
        
        print(f"Total valid detections: {len(detections)}")
        return detections
    
    def generate_mask_optimized(self, mask_coeffs, mask_protos, bbox_model_space):
        """
        Memory-optimized mask generation
        """
        try:
            
            # Compute mask with reduced precision for memory efficiency
            mask_160 = np.dot(mask_protos, mask_coeffs.astype(np.float32))
            mask_160 = 1 / (1 + np.exp(-mask_160))
            
            # Work with smaller intermediate arrays
            x1_model, y1_model, x2_model, y2_model = bbox_model_space
            x1_model = int(max(0, min(x1_model, 640)))
            y1_model = int(max(0, min(y1_model, 640)))
            x2_model = int(max(x1_model, min(x2_model, 640)))
            y2_model = int(max(y1_model, min(y2_model, 640)))
            
            # Resize only the needed portion
            mask_pil = Image.fromarray((mask_160 * 255).astype(np.uint8))
            del mask_160  # Free memory immediately
            
            # Resize in steps to reduce memory usage
            mask_320 = mask_pil.resize((320, 320), Image.LANCZOS)
            del mask_pil
            mask_640 = mask_320.resize((640, 640), Image.LANCZOS)
            del mask_320
            
            mask_640_np = np.array(mask_640, dtype=np.uint8) / 255.0
            del mask_640
            
            # Extract and resize crop
            mask_crop = mask_640_np[y1_model:y2_model, x1_model:x2_model]
            del mask_640_np
            
            if mask_crop.size == 0:
                return np.ones((10, 10), dtype=np.uint8)
            
            # Final resize to target size
            target_w = max(10, int(x2_model - x1_model))
            target_h = max(10, int(y2_model - y1_model))
            
            mask_crop_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
            del mask_crop
            mask_resized = mask_crop_pil.resize((target_w, target_h), Image.LANCZOS)
            del mask_crop_pil
            
            final_mask = np.array(mask_resized, dtype=np.uint8)
            del mask_resized
            
            # Threshold to binary
            final_mask = (final_mask > 127).astype(np.uint8)
            
            return final_mask
            
        except Exception as e:
            print(f"Error in mask generation: {e}")
            return np.ones((10, 10), dtype=np.uint8)
    
    def apply_nms(self, detections):
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            filtered_detections: List of detections after NMS
        """
        if len(detections) == 0:
            return []
            
        # Extract bounding boxes and scores
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # Simple NMS implementation using only numpy
        def compute_iou(box1, box2):
            x1_max = max(box1[0], box2[0])
            y1_max = max(box1[1], box2[1])
            x2_min = min(box1[2], box2[2])
            y2_min = min(box1[3], box2[3])
            
            if x2_min <= x1_max or y2_min <= y1_max:
                return 0.0
                
            intersection = (x2_min - x1_max) * (y2_min - y1_max)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        # Sort by confidence
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = []
        
        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
                
            current_box = boxes[current_idx]
            remaining_indices = []
            
            for idx in sorted_indices[1:]:
                if compute_iou(current_box, boxes[idx]) < self.iou_threshold:
                    remaining_indices.append(idx)
                    
            sorted_indices = np.array(remaining_indices)
        
        return [detections[i] for i in keep_indices]
    
    def predict(self, image_path):
        """
        Run inference on an image (handles both FLOAT32 and INT8 models)
        
        Args:
            image_path: Path to input image or PIL Image object
            
        Returns:
            detections: List of detection results
        """
        # Preprocess image
        preprocessed_image = self.preprocess_image(image_path)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs and dequantize if necessary
        outputs = []
        for i, output_detail in enumerate(self.output_details):
            output_data = self.interpreter.get_tensor(output_detail['index'])
            
            # Dequantize output if it's quantized
            quant_info = self.output_quantization_info[i]
            if quant_info['is_quantized']:
                scale = quant_info['scale']
                zero_point = quant_info['zero_point']
                output_data = (output_data.astype(np.float32) - zero_point) * scale
                print(f"Dequantized output {i}: scale={scale}, zero_point={zero_point}")
            
            outputs.append(output_data)
        
        # Post-process results
        detections = self.postprocess_detections(outputs)
        
        # Apply NMS
        filtered_detections = self.apply_nms(detections)
        
        return filtered_detections
    
    def visualize_masks(self, detections, image_height=640, image_width=640, save_path=None):
        """
        Generate grayscale mask image (0 = background, 255 = object)
        """
        if len(detections) == 0:
            print('No detections found')
            return None

        # Single-channel grayscale mask
        mask_overlay = np.zeros((image_height, image_width), dtype=np.uint8)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            mask = det['mask']

            print(f'Detection {i+1}: water, conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2})')

            if mask is not None and mask.size > 0:
                mask_h, mask_w = mask.shape
                end_y = min(y1 + mask_h, image_height)
                end_x = min(x1 + mask_w, image_width)
                actual_h = end_y - y1
                actual_w = end_x - x1

                if actual_h > 0 and actual_w > 0:
                    if mask_h != actual_h or mask_w != actual_w:
                        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_resized = mask_pil.resize((actual_w, actual_h), Image.LANCZOS)
                        mask = (np.array(mask_resized) > 127).astype(np.uint8)

                    # Paste mask into overlay (set detected regions to 255)
                    mask_overlay[y1:end_y, x1:end_x] = np.where(mask > 0, 255, mask_overlay[y1:end_y, x1:end_x])

                    print(f'  Mask applied to region: ({x1},{y1}) to ({end_x},{end_y})')

        result_image = Image.fromarray(mask_overlay, mode='L')  # 'L' = 8-bit grayscale
        if save_path:
            result_image.save(save_path)
        return 0


# Example usage
def main():
    # Initialize model
    
    t = time.time()
    
    model_path = "models/water_int8.tflite"
    yolo = YOLOv11Segmentation(model_path, conf_threshold=0.5)
    
    # Run inference
    
    image_path = "water.png"
    if os.path.exists("./photos"):
        image_files = sorted(os.listdir("./photos"))
        if len(image_files) > 0:
            image_path = os.path.join("./photos", image_files[-1])  # Use the latest image
    
    print(f"\nUsing image: {image_path}")
    detections = yolo.predict(image_path)
    
    # Generate and visualize segmentation masks instead of bounding boxes
    # yolo.visualize_masks(
    #     detections,
    #     save_path="water_segmentation_result.jpg"
    # )
    
    print("\nWater segmentation inference completed successfully!")
    print("Result images saved: water_segmentation_result.jpg")
    print(f"Total time: {time.time() - t:.2f} seconds")
    
    return 50

if __name__ == "__main__":
    main()