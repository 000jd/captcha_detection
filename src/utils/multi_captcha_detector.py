import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import cv2
import os
import yaml
from ultralytics import YOLO

class MultiCAPTCHADetector:
    """
    A unified solution for detecting and recognizing characters in different types of CAPTCHAs
    using a combination of YOLO object detection and image preprocessing techniques.
    """
    
    def __init__(self, 
                 model_path=None,
                 classes_file=None,
                 conf_threshold=0.25,
                 device=None):
        """
        Initialize the CAPTCHA detector
        
        Args:
            model_path: Path to the YOLO model weights
            classes_file: Path to the file containing class names
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.conf_threshold = conf_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class names
        self.class_names = self._load_class_names(classes_file)
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print("No model provided or model file not found. Only preprocessing will be available.")
            self.model = None
            
        # Import preprocessing utilities
        try:
            from captcha_detection.src.utils.captcha_transform import CAPTCHATransform
            from captcha_detection.src.utils.captcha_preprocessor import CAPTCHAPreprocessor
            self.preprocessor = CAPTCHAPreprocessor()
            self.transformer = CAPTCHATransform()
        except ImportError:
            # Fallback to local imports if package structure is not available
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.utils.captcha_transform import CAPTCHATransform
            from src.utils.captcha_preprocessor import CAPTCHAPreprocessor
            self.preprocessor = CAPTCHAPreprocessor()
            self.transformer = CAPTCHATransform()
    
    def _load_class_names(self, classes_file):
        """Load class names from file"""
        if not classes_file or not os.path.exists(classes_file):
            # Default to basic alphanumeric characters if no file provided
            return list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def preprocess_image(self, image_path, augment=False):
        """
        Preprocess an image for detection
        
        Args:
            image_path: Path to the image file or PIL Image object
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed PIL Image
        """
        # Load image if path is provided
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path
            
        # Apply augmentation if requested
        if augment:
            img = self.transformer(img)
            
        # Apply preprocessing
        img = self.preprocessor(img)
        
        return img
    
    def detect_captcha(self, image_path, preprocess=True, return_image=False):
        """
        Detect and recognize characters in a CAPTCHA image
        
        Args:
            image_path: Path to the image file or PIL Image object
            preprocess: Whether to apply preprocessing
            return_image: Whether to return the annotated image
            
        Returns:
            Dictionary with detection results and optionally the annotated image
        """
        if self.model is None:
            raise ValueError("No model loaded. Please provide a valid model path.")
            
        # Load and preprocess image
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path
            
        if preprocess:
            processed_img = self.preprocess_image(img, augment=False)
        else:
            processed_img = img
            
        # Run inference
        results = self.model(processed_img, conf=self.conf_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # Sort detections by x-coordinate for proper reading order
        detections.sort(key=lambda x: x['bbox'][0])
        
        # Extract the text
        captcha_text = ''.join([d['class'] for d in detections])
        
        result = {
            'text': captcha_text,
            'detections': detections,
            'confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
        }
        
        if return_image:
            # Draw detections on image
            annotated_img = self._draw_detections(img, detections)
            result['image'] = annotated_img
            
        return result
    
    def _draw_detections(self, img, detections):
        """Draw detection boxes and labels on the image"""
        # Convert PIL to OpenCV format
        img_cv = np.array(img)
        if len(img_cv.shape) == 2:  # Grayscale
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        else:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls = det['class']
            conf = det['confidence']
            
            # Draw box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{cls} {conf:.2f}"
            cv2.putText(img_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    def train_model(self, data_yaml, epochs=100, batch_size=16, img_size=640, save_dir='runs/train'):
        """
        Train the YOLO model on CAPTCHA dataset
        
        Args:
            data_yaml: Path to the YAML file with dataset configuration
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size for training
            save_dir: Directory to save training results
            
        Returns:
            Path to the best trained model
        """
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml}")
            
        # Initialize a new YOLO model
        model = YOLO('yolo11n.pt')
        
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=os.path.dirname(save_dir),
            name=os.path.basename(save_dir)
        )
        
        # Load the best model
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            self.model = YOLO(best_model_path)
            return best_model_path
        else:
            print(f"Warning: Best model not found at {best_model_path}")
            return None
    
    def detect_captcha_type(self, image_path):
        """
        Automatically detect the type of CAPTCHA in the image
        
        Args:
            image_path: Path to the image file or PIL Image object
            
        Returns:
            String indicating the detected CAPTCHA type
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path
            
        # Convert to numpy array
        img_np = np.array(img)
        
        # Extract features for classification
        features = {}
        
        # Check image dimensions
        features['width'] = img.width
        features['height'] = img.height
        features['aspect_ratio'] = img.width / img.height
        
        # Check color properties
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            features['is_color'] = True
            # Convert to grayscale for further analysis
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            features['is_color'] = False
            gray = img_np if len(img_np.shape) == 2 else img_np[:,:,0]
            
        # Check for lines and distortions
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        features['has_lines'] = lines is not None and len(lines) > 0
        
        # Check for noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        _, noise_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        noise_ratio = np.sum(noise_mask) / (noise_mask.shape[0] * noise_mask.shape[1] * 255)
        features['noise_level'] = noise_ratio
        
        # Check for background complexity
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['contour_count'] = len(contours)
        
        # Classify CAPTCHA type based on features
        if features['noise_level'] > 0.1 and features['has_lines']:
            captcha_type = "complex_distorted"
        elif features['is_color'] and features['noise_level'] > 0.05:
            captcha_type = "color_noisy"
        elif features['contour_count'] > 20:
            captcha_type = "character_segmented"
        elif features['aspect_ratio'] > 3:
            captcha_type = "wide_format"
        else:
            captcha_type = "simple_text"
            
        return captcha_type, features
