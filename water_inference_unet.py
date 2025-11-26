import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import time

MASK_THRESHOLD = 0.5
class u_net_model:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_scale, self.input_zero = self.input_details[0]['quantization']
        self.output_scale, self.output_zero = self.output_details[0]['quantization']
        _, self.input_h, self.input_w, self.input_c = self.input_details[0]['shape']

    def load_image(self, image_path):
        img = Image.open(image_path)

        # Ensure RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to model input size
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)

        # Convert to numpy
        img_np = np.array(img, dtype=np.float32)
        return img_np
    
    def preprocess(self, img_np):
        # Quantize image to int8
        # q = f/scale + zero_point
        img_q = img_np / 255.0               # normalize [0,1]
        img_q = img_q / self.input_scale + self.input_zero
        img_q = np.clip(img_q, -128, 127).astype(np.int8)

        # Add batch dimension
        input_tensor = np.expand_dims(img_q, axis=0)
        return input_tensor
    
    def infer(self, input_tensor):       
        # Set input tensor
  # shape: 256x256
        return output_f
    
    def save_mask(self, output_f, save_path, threshold=MASK_THRESHOLD):
        mask = (output_f > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_img.save(save_path)
        
    def calculate_percentage(self, output_f, threshold=MASK_THRESHOLD):
        mask = (output_f > threshold).astype(np.uint8).flatten()
        water_pixels = np.sum(mask)
        total_pixels = len(mask)
        percentage = (water_pixels / total_pixels) * 100
        return percentage
    
    def predict(self, image_path):
        
        t = time.time()
        img_np = self.load_image(image_path)
        input_tensor = self.preprocess(img_np)
        
        t1 = time.time()
        print(f"Preprocessing time: {t1 - t:.3f} seconds")
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)

        t2 = time.time()
        print(f"Set tensor time: {t2 - t1:.3f} seconds")
        # Run inference
        self.interpreter.invoke()

        t3 = time.time()
        print(f"Inference time: {t3 - t2:.3f} seconds")
        # Retrieve output
        output_q = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # shape: 256x256x1

        # Dequantize: f = (q - zero_point) * scale
        output_f = (output_q.astype(np.float32) - self.output_zero) * self.output_scale
        output_f = np.squeeze(output_f) 
        output_f = self.infer(input_tensor)
        
        t4 = time.time()
        print(f"Postprocessing time: {t4 - t3:.3f} seconds")
        return output_f
    
def main():
    model_path = "models/tiny_unet_int8.tflite"
    image_path = "water_720.png"

    model = u_net_model(model_path)
    output = model.predict(image_path)
    # model.save_mask(mask, save_path)
    percentage = model.calculate_percentage(output)
    
    print(f"Water coverage percentage: {percentage:.2f}%")
    return percentage
    
if __name__ == "__main__":
    main()
