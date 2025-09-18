import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import gc  # For garbage collection
import time

class YOLOv11Segmentation:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.7, 
                 max_image_size=1024):
        """
        Initialize YOLOv11 segmentation model
        
        Args:
            model_path: Path to the .tflite model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_image_size: Maximum image dimension for memory optimization
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
        
        print(f"Model loaded successfully")
        print(f"Input shape: {self.input_shape}")
        
    def preprocess_image(self, image_path):
        """
        Memory-optimized image preprocessing with original image info
        """
        # Load image and get original dimensions
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image_path.convert('RGB')
            
        orig_width, orig_height = original_image.size
        
        # Resize large images for memory efficiency
        if orig_width > self.max_image_size or orig_height > self.max_image_size:
            ratio = min(self.max_image_size / orig_width, self.max_image_size / orig_height)
            new_width = int(orig_width * ratio)
            new_height = int(orig_height * ratio)
            print(f"Resizing image from {orig_width}x{orig_height} to {new_width}x{new_height}")
            original_image = original_image.resize((new_width, new_height), Image.LANCZOS)
            orig_width, orig_height = new_width, new_height

        # Resize to model input size
        resized_image = original_image.resize((self.input_width, self.input_height), Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(resized_image, dtype=np.float32)
        del resized_image
        
        # Normalize
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Calculate scale factors for coordinate conversion
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        gc.collect()
        
        return image_array, (orig_width, orig_height), (scale_x, scale_y)
    
    def postprocess_detections(self, outputs, original_size, scale_factors):
        """
        Memory-optimized post-processing of YOLOv11 outputs
        """
        detections = []
        orig_width, orig_height = original_size
        scale_x, scale_y = scale_factors
        
        # Extract outputs
        detection_output = outputs[0][0]  # (37, 8400)
        mask_protos = outputs[1][0]  # (160, 160, 32)
        
        # Transpose detection output
        if detection_output.shape[0] < detection_output.shape[1]:
            print("Transposing detection output from (37, 8400) to (8400, 37)")
            detection_output = detection_output.T
        
        print(f"Detection output shape: {detection_output.shape}")
        
        # Process detections in smaller chunks for memory efficiency
        chunk_size = 1000  # Smaller chunks for embedded devices
        num_detections = detection_output.shape[0]
        
        for chunk_start in range(0, num_detections, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_detections)
            detection_chunk = detection_output[chunk_start:chunk_end]
            
            for i, detection in enumerate(detection_chunk):
                # Extract coordinates and confidence
                x_center, y_center, width, height = detection[0:4]
                confidence = detection[4]
                
                if confidence < self.conf_threshold:
                    continue
                
                # Convert normalized coordinates to model pixels
                x_center_px = x_center * 640
                y_center_px = y_center * 640
                width_px = width * 640
                height_px = height * 640
                
                # Convert to corner coordinates in model space
                x1_model = x_center_px - width_px/2
                y1_model = y_center_px - height_px/2
                x2_model = x_center_px + width_px/2
                y2_model = y_center_px + height_px/2
                
                # Scale to original image space
                x1 = x1_model * scale_x
                y1 = y1_model * scale_y
                x2 = x2_model * scale_x
                y2 = y2_model * scale_y
                
                # Bounds checking in original image space
                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(x1 + 1, min(x2, orig_width))
                y2 = max(y1 + 1, min(y2, orig_height))
                
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width > 10 and box_height > 10:
                    # Generate mask for valid detections
                    mask = None
                    if confidence > 0.5:
                        try:
                            mask_coeffs = detection[5:37]
                            mask = self.generate_mask_optimized(
                                mask_coeffs, mask_protos,
                                (x1_model, y1_model, x2_model, y2_model),
                                scale_factors, orig_width, orig_height
                            )
                        except Exception as e:
                            print(f"Warning: Mask generation failed: {e}")
                            mask = None
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': 0,
                        'mask': mask
                    })
            
            gc.collect()
        
        print(f"Total valid detections: {len(detections)}")
        return detections
    
    def generate_mask_optimized(self, mask_coeffs, mask_protos, bbox_model_space, 
                              scale_factors, orig_width, orig_height):
        """
        Memory-optimized mask generation with proper scaling
        """
        try:
            scale_x, scale_y = scale_factors
            
            # Compute mask
            mask_160 = np.dot(mask_protos, mask_coeffs.astype(np.float32))
            mask_160 = 1 / (1 + np.exp(-mask_160))
            
            # Get bounding box in model space
            x1_model, y1_model, x2_model, y2_model = bbox_model_space
            x1_model = int(max(0, min(x1_model, 640)))
            y1_model = int(max(0, min(y1_model, 640)))
            x2_model = int(max(x1_model + 1, min(x2_model, 640)))
            y2_model = int(max(y1_model + 1, min(y2_model, 640)))
            
            # Resize mask to model input size (640x640)
            mask_pil = Image.fromarray((mask_160 * 255).astype(np.uint8))
            del mask_160
            
            mask_640 = mask_pil.resize((640, 640), Image.LANCZOS)
            del mask_pil
            
            mask_640_np = np.array(mask_640, dtype=np.float32) / 255.0
            del mask_640
            
            # Crop mask to detection region in model space
            mask_crop = mask_640_np[y1_model:y2_model, x1_model:x2_model]
            del mask_640_np
            
            if mask_crop.size == 0:
                return np.ones((20, 20), dtype=np.uint8)
            
            # Scale to original image size
            target_w = max(20, int((x2_model - x1_model) * scale_x))
            target_h = max(20, int((y2_model - y1_model) * scale_y))
            
            # Ensure target size doesn't exceed original image bounds
            target_w = min(target_w, orig_width)
            target_h = min(target_h, orig_height)
            
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
            return np.ones((20, 20), dtype=np.uint8)
    
    def apply_nms(self, detections):
        """
        Apply Non-Maximum Suppression
        """
        if len(detections) == 0:
            return []
            
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
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
        Run inference on an image
        """
        # Preprocess image
        preprocessed_image, original_size, scale_factors = self.preprocess_image(image_path)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
        del preprocessed_image
        gc.collect()
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output_data.copy())
        
        # Post-process results
        detections = self.postprocess_detections(outputs, original_size, scale_factors)
        del outputs
        gc.collect()
        
        # Apply NMS
        filtered_detections = self.apply_nms(detections)
        
        return filtered_detections, original_size
    
    def visualize_masks(self, detections, original_size, save_path=None):
        """
        Memory-optimized mask visualization
        """
        orig_width, orig_height = original_size
        
        if len(detections) == 0:
            print("No detections found")
            return None
        
        # Create mask overlay for original image size
        mask_overlay = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
        water_color = np.array([0, 255, 255], dtype=np.uint8)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            mask = det['mask']
            
            print(f"Detection {i+1}: water, conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2})")
            
            if mask is not None and mask.size > 0:
                mask_h, mask_w = mask.shape
                end_y = min(y1 + mask_h, orig_height)
                end_x = min(x1 + mask_w, orig_width)
                actual_h = end_y - y1
                actual_w = end_x - x1
                
                if actual_h > 0 and actual_w > 0:
                    # Resize mask to fit exactly in bounding box
                    if mask_h != actual_h or mask_w != actual_w:
                        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_resized = mask_pil.resize((actual_w, actual_h), Image.LANCZOS)
                        mask = (np.array(mask_resized) > 127).astype(np.uint8)
                        del mask_pil, mask_resized
                    
                    # Apply mask to overlay
                    mask_region = mask_overlay[y1:end_y, x1:end_x]
                    for c in range(3):
                        mask_region[:, :, c] = np.where(mask > 0, water_color[c], mask_region[:, :, c])
                    
                    print(f"  Mask applied to region: ({x1},{y1}) to ({end_x},{end_y})")
                    
                    del mask
                    gc.collect()
        
        # Save result
        result_image = Image.fromarray(mask_overlay)
        if save_path:
            result_image.save(save_path)
            print(f"Result saved to: {save_path}")
        
        del mask_overlay
        gc.collect()
        
        return result_image


# Example usage
def main():
    t = time.time()
    
    model_path = "models/water_float32.tflite"
    yolo = YOLOv11Segmentation(model_path, conf_threshold=0.5, max_image_size=1024)
    
    # Run inference
    image_path = "water.jpg"
    detections, original_size = yolo.predict(image_path)
    
    # Generate mask visualization
    yolo.visualize_masks(detections, original_size, save_path="water_segmentation_result.jpg")
    
    print("Water segmentation inference completed successfully!")
    print("Result images saved: water_segmentation_result.jpg")
    print(f"Total time: {time.time() - t:.2f} seconds")

if __name__ == "__main__":
    main()