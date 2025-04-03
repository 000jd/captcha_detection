import cv2
import numpy as np
from PIL import Image, ImageOps
import os

class CAPTCHAPreprocessor:
    def __init__(self, output_size=(128,64), enhancement_level=3.0):
        self.output_size = output_size
        self.enhancement_level = enhancement_level
        
    def __call__(self, img):
        try:
            # Ensure image is in an acceptable mode
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
                
            # Convert to RGB or L mode if needed
            if img.mode not in ['RGB', 'RGBA', 'L']:
                img = img.convert('RGB')
                
            img = self._remove_background(img)
            img = self._normalize_colors(img)
            img = self._enhance_features(img)
            img = self._align_characters(img)
            return img
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Return original image if preprocessing fails
            if isinstance(img, np.ndarray):
                return Image.fromarray(img)
            return img

    def _remove_background(self, img):
        """Remove background and extract text"""
        try:
            # Convert to numpy array for OpenCV processing
            np_img = np.array(img)
            
            # Handle different color spaces
            if len(np_img.shape) == 3:
                if np_img.shape[2] == 4:  # RGBA
                    # Extract alpha channel
                    alpha = np_img[:,:,3]
                    np_img = np_img[:,:,:3]
                    # Create white background
                    bg = np.ones_like(np_img) * 255
                    # Blend based on alpha
                    np_img = np.where(alpha[:,:,None] > 128, np_img, bg)
                
                # Convert to grayscale
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = np_img
                
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 10)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask from contours
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, contours, -1, 255, -1)
            
            # Apply mask to original image
            if len(np_img.shape) == 3:
                result = cv2.bitwise_and(np_img, np_img, mask=mask)
            else:
                result = cv2.bitwise_and(np_img, np_img, mask=mask)
                
            return Image.fromarray(result)
        except Exception as e:
            print(f"Error in removing background: {e}")
            return img

    def _normalize_colors(self, img):
        """Normalize colors to improve OCR consistency"""
        try:
            # Convert to Image if needed
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
                
            # Convert to grayscale
            img = img.convert('L')
            
            # Enhance contrast
            img = ImageOps.autocontrast(img, cutoff=5)
            
            return img
        except Exception as e:
            print(f"Error in normalizing colors: {e}")
            return img

    def _enhance_features(self, img):
        """Enhance features to make characters more recognizable"""
        try:
            # Convert to numpy array
            np_img = np.array(img)
            
            # Skip if empty or invalid
            if np_img.size == 0:
                return img
                
            # Ensure grayscale
            if len(np_img.shape) == 3:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
                
            # Check if image is valid for CLAHE
            if np.min(np_img) == np.max(np_img):
                return img
                
            # CLAHE contrast enhancement
            clahe = cv2.createCLAHE(
                clipLimit=self.enhancement_level,
                tileGridSize=(8,8)
            )
            enhanced = clahe.apply(np_img)
            
            # Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            return Image.fromarray(cleaned)
        except Exception as e:
            print(f"Error in enhancing features: {e}")
            return img

    def _align_characters(self, img):
        """Center and normalize characters for consistent input"""
        try:
            # Convert to numpy array
            np_img = np.array(img)
            
            # Skip processing for invalid images
            if np_img.size == 0:
                # Return a blank image of the right size
                blank = np.zeros((self.output_size[1], self.output_size[0]), dtype=np.uint8)
                return Image.fromarray(blank)
            
            # Find contours
            if len(np_img.shape) == 3:
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = np_img
                
            # Binarize if needed
            if np.max(gray) == np.min(gray):
                binary = gray
            else:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create final image
            final = np.zeros((self.output_size[1], self.output_size[0]), dtype=np.uint8)
            
            if contours:
                # Find combined bounding box
                x, y, w, h = cv2.boundingRect(np.concatenate(contours))
                
                # Ensure we don't have zero dimensions
                if w > 0 and h > 0:
                    # Crop to the bounding box
                    cropped = gray[y:y+h, x:x+w]
                    
                    # Resize with aspect ratio preservation
                    ratio = min(self.output_size[0]/w, self.output_size[1]/h)
                    new_size = (int(w*ratio), int(h*ratio))
                    
                    resized = cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)
                    
                    # Center in output image
                    x_offset = (self.output_size[0] - new_size[0]) // 2
                    y_offset = (self.output_size[1] - new_size[1]) // 2
                    
                    final[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
            
            return Image.fromarray(final)
        except Exception as e:
            print(f"Error in aligning characters: {e}")
            # Return original image resized to output dimensions
            try:
                return img.resize(self.output_size)
            except:
                blank = np.zeros((self.output_size[1], self.output_size[0]), dtype=np.uint8)
                return Image.fromarray(blank)
