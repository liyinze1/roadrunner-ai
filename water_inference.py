import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import time

class YOLOv11Segmentation:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.7):
        """
        Initialize YOLOv11 segmentation model
        
        Args:
            model_path: Path to the .tflite model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors info
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape (typically [1, height, width, 3] for YOLOv11)
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        print(f"Model loaded successfully")
        print(f"Input shape: {self.input_shape}")
        print(f"Input dtype: {self.input_details[0]['dtype']}")
        print(f"Number of outputs: {len(self.output_details)}")
        
    def preprocess_image(self, image_path):
        """
        Preprocess input image for YOLOv11
        
        Args:
            image_path: Path to input image or PIL Image object
            
        Returns:
            preprocessed_image: Normalized image array
            original_image: Original PIL image
            scale_factor: Scaling factors for coordinate conversion
        """
        # Load image
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image_path.convert('RGB')
            
        orig_width, orig_height = original_image.size
        
        # Resize image to model input size
        resized_image = original_image.resize((self.input_width, self.input_height), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(resized_image, dtype=np.float32)
        
        # YOLOv11 typically expects input in range [0, 1]
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Calculate scale factors for coordinate conversion
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        return image_array, original_image, (scale_x, scale_y)
    
    def postprocess_detections(self, outputs, scale_factors):
        """
        Post-process YOLOv11 outputs to extract detections and masks
        
        Args:
            outputs: Model outputs
            scale_factors: Scaling factors for coordinate conversion
            
        Returns:
            detections: List of detection dictionaries
        """
        scale_x, scale_y = scale_factors
        detections = []
        
        # Debug: Print output information
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
        
        # Extract detection and mask outputs
        detection_output = outputs[0][0]  # Remove batch dimension: (37, 8400)
        mask_protos = outputs[1][0]       # Remove batch dimension: (160, 160, 32)
        
        print(f"Detection output shape: {detection_output.shape}")
        print(f"Mask prototypes shape: {mask_protos.shape}")
        
        # Transpose detection output from (37, 8400) to (8400, 37)
        if detection_output.shape[0] < detection_output.shape[1]:
            print("Transposing detection output from (37, 8400) to (8400, 37)")
            detection_output = detection_output.T
        
        print(f"After transpose - Detection output shape: {detection_output.shape}")
        
        num_detections = detection_output.shape[0]  # 8400
        num_values_per_detection = detection_output.shape[1]  # 37
        
        print(f"Number of detections: {num_detections}")
        print(f"Values per detection: {num_values_per_detection}")
        
        # YOLOv11 segmentation format: [x, y, w, h, conf, 32_mask_coefficients]
        # Total: 4 + 1 + 32 = 37 values per detection
        
        for i, detection in enumerate(detection_output):
            if num_values_per_detection >= 37:
                # Extract coordinates, confidence, and mask coefficients
                x_center, y_center, width, height = detection[0:4]
                confidence = detection[4]
                mask_coeffs = detection[5:37]  # 32 mask coefficients
                
                # Debug first few detections
                if i < 3 and confidence > 0.1:
                    print(f"Detection {i}: x={x_center:.3f}, y={y_center:.3f}, w={width:.3f}, h={height:.3f}, conf={confidence:.3f}")
                    print(f"  First few mask coeffs: {mask_coeffs[:5]}")
                
                if confidence < self.conf_threshold:
                    continue
                
                # Convert from normalized coordinates to pixel coordinates
                x_center_px = x_center * 640  # Model input size
                y_center_px = y_center * 640
                width_px = width * 640
                height_px = height * 640
                
                # Convert center format to corner format in model space
                x1_model = x_center_px - width_px/2
                y1_model = y_center_px - height_px/2
                x2_model = x_center_px + width_px/2
                y2_model = y_center_px + height_px/2
                
                # Scale to original image space
                x1 = x1_model * scale_x
                y1 = y1_model * scale_y
                x2 = x2_model * scale_x
                y2 = y2_model * scale_y
                
                # Generate mask using mask coefficients and prototypes
                mask = self.generate_mask(mask_coeffs, mask_protos, 
                                        (x1_model, y1_model, x2_model, y2_model), 
                                        scale_factors)
                
                # Ensure coordinates are valid
                x1 = max(0, min(x1, scale_x * 640))
                y1 = max(0, min(y1, scale_y * 640))
                x2 = max(x1 + 1, min(x2, scale_x * 640))
                y2 = max(y1 + 1, min(y2, scale_y * 640))
                
                # Only keep detections with reasonable size
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width > 10 and box_height > 10:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': 0,  # Single class: water
                        'mask': mask
                    })
        
        print(f"Total valid detections after filtering: {len(detections)}")
        return detections
    
    def generate_mask(self, mask_coeffs, mask_protos, bbox_model_space, scale_factors):
        """
        Generate segmentation mask from coefficients and prototypes
        
        Args:
            mask_coeffs: Mask coefficients from detection (32 values)
            mask_protos: Mask prototypes (160, 160, 32)
            bbox_model_space: Bounding box in model coordinate space
            scale_factors: Scale factors for resizing
            
        Returns:
            mask: Binary mask in original image size
        """
        scale_x, scale_y = scale_factors
        
        # Compute mask by multiplying coefficients with prototypes
        # mask_protos: (160, 160, 32), mask_coeffs: (32,)
        mask_160 = np.dot(mask_protos, mask_coeffs)  # Result: (160, 160)
        
        # Apply sigmoid activation to get probabilities
        mask_160 = 1 / (1 + np.exp(-mask_160))
        
        # Resize mask from 160x160 to 640x640 (model input size)
        from PIL import Image
        mask_pil = Image.fromarray((mask_160 * 255).astype(np.uint8))
        mask_640 = mask_pil.resize((640, 640), Image.LANCZOS)
        mask_640_np = np.array(mask_640) / 255.0
        
        # Crop mask to bounding box region in model space
        x1_model, y1_model, x2_model, y2_model = bbox_model_space
        x1_model = int(max(0, min(x1_model, 640)))
        y1_model = int(max(0, min(y1_model, 640)))
        x2_model = int(max(x1_model, min(x2_model, 640)))
        y2_model = int(max(y1_model, min(y2_model, 640)))
        
        # Extract mask region
        mask_crop = mask_640_np[y1_model:y2_model, x1_model:x2_model]
        
        if mask_crop.size == 0:
            # If crop is empty, return small dummy mask
            mask_crop = np.ones((10, 10))
        
        # Resize cropped mask to original image bounding box size
        crop_h, crop_w = mask_crop.shape
        target_w = int((x2_model - x1_model) * scale_x)
        target_h = int((y2_model - y1_model) * scale_y)
        
        if target_w > 0 and target_h > 0:
            mask_crop_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
            mask_resized = mask_crop_pil.resize((target_w, target_h), Image.LANCZOS)
            final_mask = np.array(mask_resized) / 255.0
        else:
            final_mask = mask_crop
        
        # Threshold to get binary mask
        final_mask = (final_mask > 0.5).astype(np.uint8)
        
        return final_mask
    
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
        Run inference on an image
        
        Args:
            image_path: Path to input image or PIL Image object
            
        Returns:
            detections: List of detection results
            original_image: Original input image
        """
        # Preprocess image
        preprocessed_image, original_image, scale_factors = self.preprocess_image(image_path)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output_data)
        
        # Post-process results
        detections = self.postprocess_detections(outputs, scale_factors)
        
        # Apply NMS
        filtered_detections = self.apply_nms(detections)
        
        return filtered_detections, original_image
    
    def visualize_masks(self, detections, image, class_names=None, save_path=None, save_mask_only=False):
        """
        Visualize segmentation masks on image using PIL
        
        Args:
            detections: List of detection results with masks
            image: PIL Image
            class_names: List of class names (optional)
            save_path: Path to save the result image (optional)
            save_mask_only: If True, save only the mask without original image
            
        Returns:
            result_image: PIL Image with mask visualizations
        """
        print(f"Original image size: {image.size}")
        
        if len(detections) == 0:
            print("No detections found - saving original image")
            if save_path:
                image.save(save_path)
            return image
        
        # Create mask overlay
        mask_overlay = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        
        print(f"Found {len(detections)} detections with masks:")
        
        # Water color: bright cyan/blue
        water_color = np.array([0, 255, 255], dtype=np.uint8)  # Cyan for water
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            mask = det['mask']
            class_name = class_names[0] if class_names else "water"
            
            print(f"Detection {i+1}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {conf:.3f}")
            print(f"  BBox: ({x1}, {y1}, {x2}, {y2})")
            print(f"  Mask shape: {mask.shape if mask is not None else 'None'}")
            
            if mask is not None and mask.size > 0:
                # Place mask in the correct position on the full image
                mask_h, mask_w = mask.shape
                
                # Ensure mask fits within bounding box
                end_y = min(y1 + mask_h, image.height)
                end_x = min(x1 + mask_w, image.width)
                actual_h = end_y - y1
                actual_w = end_x - x1
                
                if actual_h > 0 and actual_w > 0:
                    # Resize mask if needed to fit exactly in bounding box
                    if mask_h != actual_h or mask_w != actual_w:
                        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_resized = mask_pil.resize((actual_w, actual_h), Image.LANCZOS)
                        mask = np.array(mask_resized) / 255.0
                        mask = (mask > 0.5).astype(np.uint8)
                    
                    # Apply mask to overlay
                    mask_region = mask_overlay[y1:end_y, x1:end_x]
                    for c in range(3):
                        mask_region[:, :, c] = np.where(mask > 0, water_color[c], mask_region[:, :, c])
                    
                    print(f"  Mask applied to region: ({x1},{y1}) to ({end_x},{end_y})")
                else:
                    print(f"  Warning: Invalid mask region")
            else:
                print(f"  Warning: No valid mask found")
        
        if save_mask_only:
            # Save only the mask
            result_image = Image.fromarray(mask_overlay)
        else:
            # Blend mask with original image
            original_array = np.array(image)
            
            # Create alpha blending (50% transparency)
            alpha = 0.5
            mask_pixels = np.any(mask_overlay > 0, axis=2)  # Where mask exists
            
            blended = original_array.copy()
            blended[mask_pixels] = (alpha * original_array[mask_pixels] + 
                                  (1 - alpha) * mask_overlay[mask_pixels]).astype(np.uint8)
            
            result_image = Image.fromarray(blended)
        
        # Save image if path provided
        if save_path:
            result_image.save(save_path)
            print(f"Result saved to: {save_path}")
            
            # Also save mask-only version
            if not save_mask_only:
                mask_only_path = save_path.replace('.jpg', '_mask_only.jpg')
                mask_only_image = Image.fromarray(mask_overlay)
                mask_only_image.save(mask_only_path)
                print(f"Mask-only version saved to: {mask_only_path}")
        
        return result_image


# Example usage
def main():
    # Initialize model
    now = time.time()
    
    model_path = "models/water_float32.tflite"  # Update this path
    yolo = YOLOv11Segmentation(model_path, conf_threshold=0.5)
    
    # Run inference
    image_path = "water.jpg"  # Update this path
    detections, original_image = yolo.predict(image_path)
    
    # Generate and visualize segmentation masks instead of bounding boxes
    class_names = ['water']
    
    # This will create both blended image and mask-only versions
    result_image = yolo.visualize_masks(
        detections, 
        original_image, 
        class_names, 
        save_path="water_segmentation_result.jpg"
    )
    
    print("Water segmentation inference completed successfully!")
    print("Result images saved:")
    print("- 'water_segmentation_result.jpg' (blended with original)")
    print("- 'water_segmentation_result_mask_only.jpg' (mask only)")
    
    print(f"Total time: {time.time() - now:.2f} seconds")
    
if __name__ == "__main__":
    main()