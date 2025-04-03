import torch
import cv2
import numpy as np
import random
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
from scipy.ndimage import gaussian_filter

class CAPTCHATransform:
    def __init__(self, background_dir=None, 
                 color_prob=0.2, 
                 noise_prob=0.3, 
                 warp_prob=0.3,
                 max_warp=0.2):
        """
        Args:
            background_dir: Path to background images
            color_prob: Probability of applying color distortion (0-1)
            noise_prob: Probability of adding noise (0-1)
            warp_prob: Probability of geometric distortion (0-1)
            max_warp: Maximum warping intensity (0-1)
        """
        self.backgrounds = self._load_backgrounds(background_dir) if background_dir else None
        self.color_prob = color_prob
        self.noise_prob = noise_prob
        self.warp_prob = warp_prob
        self.max_warp = max_warp

    def _load_backgrounds(self, dir_path):
        """Load background images from directory"""
        backgrounds = []
        if dir_path and os.path.exists(dir_path):
            for fname in os.listdir(dir_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = Image.open(os.path.join(dir_path, fname)).convert('RGB')
                        backgrounds.append(img)
                    except Exception as e:
                        print(f"Error loading background image {fname}: {e}")
        return backgrounds
    
    def __call__(self, img):
        try:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
                
            # Ensure image is in RGB mode
            if img.mode not in ['RGB', 'RGBA', 'L']:
                img = img.convert('RGB')
                
            # Apply augmentations with configurable probabilities
            if random.random() < self.color_prob:
                img = self._color_distortion(img)
                
            if random.random() < self.noise_prob:
                img = self._add_complex_noise(img)
                
            if random.random() < self.warp_prob:
                img = self._geometric_distortion(img)
                
            if self.backgrounds and random.random() > 0.4:
                img = self._change_background(img)
                
            return img
        except Exception as e:
            print(f"Error in CAPTCHATransform.__call__: {e}")
            # Return original image if transformation fails
            return img

    def _color_distortion(self, img):
        """Apply various color transformations"""
        try:
            # Convert to RGB if not already (this is critical)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Random color adjustments
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.5, 1.5))
            
            # Random brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))
            
            # Random contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))
            
            # Random color mode changes - FIX: Ensure we convert back to RGB
            if random.random() > 0.5:
                # Create HSV-like effect without actually changing the mode
                # This avoids the PNG saving issue with HSV mode
                np_img = np.array(img)
                
                # Ensure image is in the correct format for OpenCV
                if len(np_img.shape) == 2:
                    # Convert grayscale to RGB
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
                elif np_img.shape[2] == 4:
                    # Convert RGBA to RGB
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
                
                # Convert to HSV
                try:
                    hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
                    
                    # Split HSV channels
                    h, s, v = cv2.split(hsv_img)
                    
                    # Apply random adjustments to HSV channels
                    # FIX: Ensure all channels have the same size and depth
                    h = h.astype(np.uint8)
                    h = (h + random.randint(0, 180)) % 180  # Hue shift
                    
                    s = s.astype(np.uint8)
                    s = np.clip(s * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
                    
                    v = v.astype(np.uint8)
                    v = np.clip(v * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
                    
                    # Verify all channels have the same shape and depth before merging
                    if h.shape == s.shape == v.shape and h.dtype == s.dtype == v.dtype:
                        # Merge channels and convert back to RGB
                        hsv_img = cv2.merge([h, s, v])
                        rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
                        
                        # Convert back to PIL Image
                        img = Image.fromarray(rgb_img)
                except Exception as e:
                    print(f"HSV conversion error: {e}")
                    # If HSV conversion fails, use alternative color adjustment
                    r, g, b = cv2.split(np_img)
                    
                    # Apply random adjustments to RGB channels
                    r = np.clip(r * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
                    g = np.clip(g * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
                    b = np.clip(b * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
                    
                    # Merge RGB channels
                    rgb_img = cv2.merge([r, g, b])
                    
                    # Convert back to PIL Image
                    img = Image.fromarray(rgb_img)
            
            return img
        except Exception as e:
            print(f"Error in color distortion: {e}")
            # Return original image if transformation fails
            return img
    
    def _geometric_distortion(self, img):
        """Apply geometric transformations like warping and perspective shifts"""
        try:
            # Convert to numpy array
            np_img = np.array(img)
            
            # Ensure image is in the correct format for OpenCV
            if len(np_img.shape) == 2:
                # Convert grayscale to RGB
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
            elif np_img.shape[2] == 4:
                # Convert RGBA to RGB
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
                
            h, w = np_img.shape[:2]
            
            # Elastic deformation
            alpha = w * random.uniform(0.5, 1.5)
            sigma = w * random.uniform(0.04, 0.08)
            
            # Create random displacement fields
            dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
            
            # Create coordinate grid
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Apply displacement fields
            map_x = grid_x + dx
            map_y = grid_y + dy
            
            # Ensure maps are within bounds
            map_x = np.clip(map_x, 0, w-1)
            map_y = np.clip(map_y, 0, h-1)
            
            # Apply remapping
            distorted = cv2.remap(np_img, map_x.astype(np.float32), map_y.astype(np.float32), 
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            return Image.fromarray(distorted)
        except Exception as e:
            print(f"Error in geometric distortion: {e}")
            # Return original image if transformation fails
            return img

    def _add_complex_noise(self, img):
        """Add various noise patterns like lines, dots, and distortions"""
        try:
            # Convert to numpy array
            np_img = np.array(img)
            
            # Ensure image is in the correct format for OpenCV
            if len(np_img.shape) == 2:
                # Convert grayscale to RGB
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
                channels = 3
            elif len(np_img.shape) == 3 and np_img.shape[2] == 4:
                # Convert RGBA to RGB
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
                channels = 3
            elif len(np_img.shape) == 3:
                channels = np_img.shape[2]
            else:
                # Fallback for unusual formats
                return img
                
            h, w = np_img.shape[:2]
            
            # Random lines
            if random.random() > 0.5:
                for _ in range(random.randint(1,5)):
                    # For color images, color is a tuple of integers
                    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    
                    # Draw line with proper color format
                    cv2.line(np_img, 
                            (random.randint(0,w), random.randint(0,h)),
                            (random.randint(0,w), random.randint(0,h)), 
                            color, random.randint(1,2))

            # Random dots
            if random.random() > 0.5:
                noise = np.random.randint(0,255,(h,w,channels), dtype=np.uint8)
                mask = np.random.rand(h,w) < 0.05
                np_img = np.where(mask[...,None], noise, np_img)

            # Wave distortion
            if random.random() > 0.7:
                np_img = cv2.warpAffine(np_img, np.float32([[1, 0.1*random.random(),0],
                                                    [0.1*random.random(),1,0]]), (w,h))

            return Image.fromarray(np_img)
        except Exception as e:
            print(f"Error in adding complex noise: {e}")
            # Return original image if transformation fails
            return img

    def _change_background(self, img):
        """Blend the CAPTCHA with a random background image"""
        try:
            if not self.backgrounds:
                return img
                
            # Select random background and resize to match image
            bg = random.choice(self.backgrounds).resize(img.size)
            
            # Create mask from original image
            if img.mode == 'L':
                mask = img
            else:
                mask = img.convert('L')
                
            # Invert and blur the mask for smoother blending
            mask = ImageOps.invert(mask).filter(ImageFilter.GaussianBlur(2))
            
            # Composite the image with background
            return Image.composite(img, bg, mask)
        except Exception as e:
            print(f"Error in changing background: {e}")
            # Return original image if transformation fails
            return img