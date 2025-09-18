import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite

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
        
        # YOLOv11 segmentation typically has multiple outputs:
        # Output 0: Detection results [batch, num_detections, 6] for single class
        # Output 1: Segmentation masks or coefficients
        
        # Use the first output for detections
        detection_output = outputs[0][0]  # Remove batch dimension
        print(f"Detection output shape: {detection_output.shape}")
        print(f"First few detections raw data:")
        print(detection_output[:5])  # Print first 5 detections for debugging
        
        # YOLOv11 output might be in format: [x_center, y_center, width, height, confidence, class]
        # or [x1, y1, x2, y2, confidence, class] depending on the model
        
        num_detections = detection_output.shape[0]
        num_values_per_detection = detection_output.shape[1] if len(detection_output.shape) > 1 else len(detection_output)
        
        print(f"Number of detections: {num_detections}")
        print(f"Values per detection: {num_values_per_detection}")
        
        # Handle different output formats
        if len(detection_output.shape) == 1:
            # Single detection format
            detection_output = detection_output.reshape(1, -1)
        
        for i, detection in enumerate(detection_output):
            if len(detection) < 5:
                continue
                
            # Try different coordinate interpretations
            if num_values_per_detection >= 6:
                # Format: [x_center, y_center, width, height, confidence, class]
                x_center, y_center, width, height, confidence = detection[0:5]
                class_id = int(detection[5]) if len(detection) > 5 else 0
                
                # Convert center format to corner format
                x1 = (x_center - width/2) * scale_x
                y1 = (y_center - height/2) * scale_y
                x2 = (x_center + width/2) * scale_x
                y2 = (y_center + height/2) * scale_y
                
            elif num_values_per_detection == 5:
                # Format: [x1, y1, x2, y2, confidence] or [x_center, y_center, width, height, confidence]
                # Try to determine which format by looking at values
                if detection[2] < detection[0] or detection[3] < detection[1]:
                    # Likely center format (width/height smaller than center coordinates)
                    x_center, y_center, width, height, confidence = detection[0:5]
                    x1 = (x_center - width/2) * scale_x
                    y1 = (y_center - height/2) * scale_y
                    x2 = (x_center + width/2) * scale_x
                    y2 = (y_center + height/2) * scale_y
                else:
                    # Likely corner format
                    x1, y1, x2, y2, confidence = detection[0:5]
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                
                class_id = 0  # Single class model
            else:
                continue
            
            # Debug print for first few detections
            if i < 3:
                print(f"Detection {i}: raw={detection[:6]}")
                print(f"  Parsed: conf={confidence:.3f}, class={class_id}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
            
            if confidence < self.conf_threshold:
                continue
            
            # Ensure coordinates are valid
            x1, y1, x2, y2 = max(0, x1), max(0, y1), max(x1+1, x2), max(y1+1, y2)
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'mask': None
            })
        
        return detections
    
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
    
    def visualize_results(self, detections, image, class_names=None, save_path=None):
        """
        Visualize detection results on image using PIL
        
        Args:
            detections: List of detection results
            image: PIL Image
            class_names: List of class names (optional)
            save_path: Path to save the result image (optional)
            
        Returns:
            result_image: PIL Image with visualizations
        """
        # Create a copy of the image for drawing
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        print(f"Original image size: {image.size}")
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        # Use a bright, easily visible color for water detection
        water_color = (0, 255, 255)  # Bright cyan for water
        
        print(f"Found {len(detections)} detections:")
        
        if len(detections) == 0:
            print("No detections found - saving original image")
            if save_path:
                result_image.save(save_path)
            return result_image
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_id = det['class_id']
            
            print(f"Detection {i+1}: bbox=({x1},{y1},{x2},{y2}), size=({x2-x1}x{y2-y1})")
            
            # Get class name
            class_name = class_names[0] if class_names else "water"  # Force water for single class
            
            # Ensure minimum box size for visibility
            box_width = max(10, x2 - x1)
            box_height = max(10, y2 - y1)
            
            # Expand small boxes to make them visible
            if box_width < 20 or box_height < 20:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                x1 = max(0, center_x - 25)
                y1 = max(0, center_y - 25)
                x2 = min(image.width, center_x + 25)
                y2 = min(image.height, center_y + 25)
                print(f"  Expanded tiny box to: ({x1},{y1},{x2},{y2})")
            
            # Draw thick bounding box for better visibility
            line_width = 4
            for offset in range(line_width):
                draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], 
                             outline=water_color, width=1)
            
            # Prepare label text
            label = f"{class_name}: {conf:.2f}"
            
            # Get text size
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width = len(label) * 10
                text_height = 20
            
            # Draw label background
            label_bg = [x1, y1 - text_height - 8, x1 + text_width + 8, y1]
            draw.rectangle(label_bg, fill=water_color, outline=water_color)
            
            # Draw label text
            draw.text((x1 + 4, y1 - text_height - 4), label, fill=(0, 0, 0), font=font)
            
            # Print detection info
            print(f"Detection {i+1}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {conf:.3f}")
            print(f"  BBox: ({x1}, {y1}, {x2}, {y2})")
            print()
        
        # Draw a border around the entire image to confirm saving worked
        draw.rectangle([0, 0, image.width-1, image.height-1], outline=(255, 0, 0), width=2)
        
        # Save image if path provided
        if save_path:
            result_image.save(save_path)
            print(f"Result saved to: {save_path}")
        
        return result_image


# Example usage
def main():
    # Initialize model
    model_path = "models/water_float32.tflite"  # Update this path
    yolo = YOLOv11Segmentation(model_path, conf_threshold=0.5)
    
    # Run inference
    image_path = "water.jpg"  # Update this path
    detections, original_image = yolo.predict(image_path)
    
    # Visualize results with PIL drawing
    # Your model has only one class: "water"
    class_names = ['water']
    
    result_image = yolo.visualize_results(detections, original_image, class_names, save_path="water_detection_output.jpg")
    
    print("Water detection inference completed successfully!")
    print("Result image saved as 'water_detection_output.jpg'")

if __name__ == "__main__":
    main()