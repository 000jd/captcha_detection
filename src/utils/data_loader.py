import os
import sys
import glob
import shutil
import random
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.captcha_transform import CAPTCHATransform
from utils.captcha_preprocessor import CAPTCHAPreprocessor

class DataLoader:
    """
    Handles loading and preprocessing of CAPTCHA images from various sources
    for training, validation, and testing.
    """
    
    def __init__(self, 
                 input_dir=None,
                 output_dir=None,
                 preprocess=True,
                 augment=False):
        """
        Initialize the data loader
        
        Args:
            input_dir: Directory containing CAPTCHA images
            output_dir: Directory to save processed images
            preprocess: Whether to apply preprocessing
            augment: Whether to apply augmentation
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.preprocess = preprocess
        self.augment = augment
        
        # Initialize transformers
        self.transformer = CAPTCHATransform() if augment else None
        self.preprocessor = CAPTCHAPreprocessor() if preprocess else None
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def load_images(self, pattern='*.png', max_images=None, return_paths=False):
        """
        Load images from input directory
        
        Args:
            pattern: File pattern to match
            max_images: Maximum number of images to load
            return_paths: Whether to return image paths along with images
            
        Returns:
            List of loaded images (and optionally paths)
        """
        if not self.input_dir:
            raise ValueError("Input directory not specified")
            
        # Find all matching files
        image_paths = glob.glob(os.path.join(self.input_dir, pattern))
        
        # Limit number of images if specified
        if max_images:
            image_paths = image_paths[:max_images]
            
        # Load images
        images = []
        valid_paths = []
        
        for img_path in tqdm(image_paths, desc="Loading images"):
            try:
                img = Image.open(img_path)
                images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if return_paths:
            return images, valid_paths
        else:
            return images
    
    def process_images(self, images, save=False, base_names=None):
        """
        Process images with augmentation and preprocessing
        
        Args:
            images: List of images to process
            save: Whether to save processed images
            base_names: Base names for saved images
            
        Returns:
            List of processed images
        """
        processed_images = []
        
        for i, img in enumerate(tqdm(images, desc="Processing images")):
            try:
                # Apply augmentation if enabled
                if self.augment and self.transformer:
                    img = self.transformer(img)
                
                # Apply preprocessing if enabled
                if self.preprocess and self.preprocessor:
                    img = self.preprocessor(img)
                
                processed_images.append(img)
                
                # Save processed image if requested
                if save and self.output_dir:
                    if base_names and i < len(base_names):
                        base_name = base_names[i]
                    else:
                        base_name = f"image_{i:04d}"
                    
                    img.save(os.path.join(self.output_dir, f"{base_name}.png"))
            except Exception as e:
                print(f"Error processing image {i}: {e}")
        
        return processed_images
    
    def load_and_process_directory(self, pattern='*.png', max_images=None, save=True):
        """
        Load and process all images in the input directory
        
        Args:
            pattern: File pattern to match
            max_images: Maximum number of images to load
            save: Whether to save processed images
            
        Returns:
            List of processed images and their paths
        """
        # Load images
        images, paths = self.load_images(pattern, max_images, return_paths=True)
        
        # Extract base names for saving
        base_names = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        
        # Process images
        processed_images = self.process_images(images, save, base_names)
        
        return processed_images, paths
    
    def convert_formats(self, input_pattern='*.jpg', output_format='png'):
        """
        Convert images from one format to another
        
        Args:
            input_pattern: File pattern to match
            output_format: Output format (e.g., 'png', 'jpg')
            
        Returns:
            Number of converted images
        """
        if not self.input_dir or not self.output_dir:
            raise ValueError("Input and output directories must be specified")
            
        # Find all matching files
        image_paths = glob.glob(os.path.join(self.input_dir, input_pattern))
        
        converted = 0
        for img_path in tqdm(image_paths, desc=f"Converting to {output_format}"):
            try:
                # Load image
                img = Image.open(img_path)
                
                # Get base name
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Save in new format
                output_path = os.path.join(self.output_dir, f"{base_name}.{output_format}")
                img.save(output_path)
                
                converted += 1
            except Exception as e:
                print(f"Error converting {img_path}: {e}")
        
        return converted
    
    def extract_characters(self, input_pattern='*.png', min_contour_area=10, max_contour_area=500):
        """
        Extract individual characters from CAPTCHA images
        
        Args:
            input_pattern: File pattern to match
            min_contour_area: Minimum contour area to consider
            max_contour_area: Maximum contour area to consider
            
        Returns:
            Number of extracted characters
        """
        if not self.input_dir or not self.output_dir:
            raise ValueError("Input and output directories must be specified")
            
        # Create characters directory
        chars_dir = os.path.join(self.output_dir, 'characters')
        os.makedirs(chars_dir, exist_ok=True)
        
        # Find all matching files
        image_paths = glob.glob(os.path.join(self.input_dir, input_pattern))
        
        extracted = 0
        for img_path in tqdm(image_paths, desc="Extracting characters"):
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Threshold
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Get base name
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Extract each character
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if min_contour_area <= area <= max_contour_area:
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Extract character
                        char_img = img[y:y+h, x:x+w]
                        
                        # Save character
                        char_path = os.path.join(chars_dir, f"{base_name}_char_{i}.png")
                        cv2.imwrite(char_path, char_img)
                        
                        extracted += 1
            except Exception as e:
                print(f"Error extracting characters from {img_path}: {e}")
        
        return extracted
    
    def create_balanced_dataset(self, class_dirs, output_dir, samples_per_class=100):
        """
        Create a balanced dataset with equal samples per class
        
        Args:
            class_dirs: Dictionary mapping class names to directories
            output_dir: Output directory for balanced dataset
            samples_per_class: Number of samples per class
            
        Returns:
            Dictionary with dataset statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {}
        for class_name, class_dir in class_dirs.items():
            try:
                # Create class directory in output
                class_output_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_output_dir, exist_ok=True)
                
                # Find all images for this class
                image_paths = glob.glob(os.path.join(class_dir, '*.png')) + \
                              glob.glob(os.path.join(class_dir, '*.jpg'))
                
                # Randomly select samples
                if len(image_paths) <= samples_per_class:
                    selected_paths = image_paths
                else:
                    selected_paths = random.sample(image_paths, samples_per_class)
                
                # Copy selected images
                for img_path in selected_paths:
                    shutil.copy(img_path, class_output_dir)
                
                stats[class_name] = len(selected_paths)
            except Exception as e:
                print(f"Error processing class {class_name}: {e}")
                stats[class_name] = 0
        
        return stats
    
    def generate_synthetic_data(self, class_names, samples_per_class=100, bg_color=(255, 255, 255), 
                               text_color=(0, 0, 0), font_size=24, noise_level=0.1):
        """
        Generate synthetic CAPTCHA data
        
        Args:
            class_names: List of class names (characters)
            samples_per_class: Number of samples per class
            bg_color: Background color
            text_color: Text color
            font_size: Font size
            noise_level: Noise level (0-1)
            
        Returns:
            Number of generated images
        """
        if not self.output_dir:
            raise ValueError("Output directory must be specified")
            
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            print("PIL.ImageDraw and PIL.ImageFont are required for synthetic data generation")
            return 0
        
        # Create synthetic data directory
        synth_dir = os.path.join(self.output_dir, 'synthetic')
        os.makedirs(synth_dir, exist_ok=True)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        generated = 0
        for class_name in tqdm(class_names, desc="Generating synthetic data"):
            # Create class directory
            class_dir = os.path.join(synth_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(samples_per_class):
                try:
                    # Create image
                    img = Image.new('RGB', (32, 32), color=bg_color)
                    draw = ImageDraw.Draw(img)
                    
                    # Calculate text position to center it
                    text_width, text_height = draw.textsize(class_name, font=font)
                    position = ((32 - text_width) // 2, (32 - text_height) // 2)
                    
                    # Draw text
                    draw.text(position, class_name, fill=text_color, font=font)
                    
                    # Add noise if requested
                    if noise_level > 0:
                        img_array = np.array(img)
                        noise = np.random.randint(0, 255, img_array.shape, dtype=np.uint8)
                        mask = np.random.rand(*img_array.shape[:2]) < noise_level
                        img_array = np.where(mask[:,:,np.newaxis], noise, img_array)
                        img = Image.fromarray(img_array)
                    
                    # Apply augmentation if enabled
                    if self.augment and self.transformer:
                        img = self.transformer(img)
                    
                    # Save image
                    img_path = os.path.join(class_dir, f"{class_name}_{i:04d}.png")
                    img.save(img_path)
                    
                    generated += 1
                except Exception as e:
                    print(f"Error generating synthetic data for {class_name}: {e}")
        
        return generated

