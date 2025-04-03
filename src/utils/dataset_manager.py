import os
import sys
import traceback
import glob
import random
import shutil
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import platform
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('DatasetManager')

class DatasetManager:
    """
    Manages dataset preparation for CAPTCHA detection training and testing.
    Handles data splitting, augmentation, and YOLO format conversion.
    """
    
    def __init__(self, 
                 input_dir=None,
                 output_dir=None,
                 train_ratio=0.8,
                 val_ratio=0.1,
                 test_ratio=0.1):
        """
        Initialize the dataset manager
        
        Args:
            input_dir: Directory containing original images and labels
            output_dir: Directory to save the processed dataset
            train_ratio: Ratio of images for training set
            val_ratio: Ratio of images for validation set
            test_ratio: Ratio of images for test set
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Detect system information for automatic configuration
        self.system_info = {
            'os': platform.system(),
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'cwd': os.getcwd()
        }
        
        logger.info(f"System detected: {self.system_info['os']} ({self.system_info['platform']})")
        
        # Import utilities if available
        self.transformer = None
        self.preprocessor = None
        try:
            from captcha_detection.utils.captcha_transform import CAPTCHATransform
            from captcha_detection.utils.captcha_preprocessor import CAPTCHAPreprocessor
            self.transformer = CAPTCHATransform()
            self.preprocessor = CAPTCHAPreprocessor()
            logger.info("Loaded CAPTCHATransform and CAPTCHAPreprocessor from captcha_detection package")
        except ImportError:
            try:
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from utils.captcha_transform import CAPTCHATransform
                from utils.captcha_preprocessor import CAPTCHAPreprocessor
                self.transformer = CAPTCHATransform()
                self.preprocessor = CAPTCHAPreprocessor()
                logger.info("Loaded CAPTCHATransform and CAPTCHAPreprocessor from local utils")
            except ImportError:
                logger.warning("Could not import CAPTCHATransform and CAPTCHAPreprocessor")
    
    def prepare_dataset(self, augment=True, num_augmentations=5):
        """
        Prepare the dataset for training, validation, and testing
        
        Args:
            augment: Whether to apply augmentation
            num_augmentations: Number of augmented versions to create per original image
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            # Validate input and output directories
            self._validate_directories()
            
            # Create output directories
            self._create_directories()
            
            # Find all images and labels
            images_dir = os.path.join(self.input_dir, 'images')
            labels_dir = os.path.join(self.input_dir, 'labels')
            
            # Get all image paths
            image_paths = self._find_images(images_dir)
            
            if not image_paths:
                raise ValueError(f"No images found in {images_dir}")
                
            logger.info(f"Found {len(image_paths)} images")
            
            # Shuffle and split
            random.shuffle(image_paths)
            
            # Calculate split indices
            num_images = len(image_paths)
            train_end = int(num_images * self.train_ratio)
            val_end = train_end + int(num_images * self.val_ratio)
            
            # Split the images
            train_images = image_paths[:train_end]
            val_images = image_paths[train_end:val_end]
            test_images = image_paths[val_end:]
            
            logger.info(f"Split: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")
            
            # Process each split
            train_stats = self._process_split(train_images, 'train', labels_dir, augment, num_augmentations)
            val_stats = self._process_split(val_images, 'val', labels_dir, False, 0)
            test_stats = self._process_split(test_images, 'test', labels_dir, False, 0)
            
            # Create dataset YAML file
            yaml_path = self._create_dataset_yaml()
            
            return {
                'train': train_stats,
                'val': val_stats,
                'test': test_stats,
                'yaml_path': yaml_path,
                'total_images': len(train_images) + len(val_images) + len(test_images),
                'augmented_images': train_stats['augmented'] if augment else 0
            }
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            logger.debug(traceback.format_exc())
            # Create a fallback dataset YAML if possible
            if hasattr(self, 'output_dir') and self.output_dir and os.path.exists(self.output_dir):
                try:
                    fallback_yaml = self._create_fallback_yaml()
                    logger.info(f"Created fallback dataset YAML: {fallback_yaml}")
                    return {
                        'error': str(e),
                        'yaml_path': fallback_yaml
                    }
                except Exception as yaml_error:
                    logger.error(f"Failed to create fallback YAML: {str(yaml_error)}")
            raise
    
    def _validate_directories(self):
        """Validate input and output directories"""
        if not self.input_dir:
            raise ValueError("Input directory not specified")
            
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory not found: {self.input_dir}")
            
        if not self.output_dir:
            raise ValueError("Output directory not specified")
            
        # Check for images and labels directories
        images_dir = os.path.join(self.input_dir, 'images')
        labels_dir = os.path.join(self.input_dir, 'labels')
        
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
            
        if not os.path.exists(labels_dir):
            raise ValueError(f"Labels directory not found: {labels_dir}")
    
    def _find_images(self, images_dir):
        """Find all images in the directory"""
        image_paths = []
        
        # Try different image extensions
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            found = glob.glob(os.path.join(images_dir, ext))
            image_paths.extend(found)
            
        # Check if we found any images
        if not image_paths:
            logger.warning(f"No images found in {images_dir}")
            
            # Check if directory exists but is empty
            if os.path.exists(images_dir) and os.path.isdir(images_dir):
                if not os.listdir(images_dir):
                    logger.error(f"Images directory {images_dir} is empty")
                else:
                    # Directory has files but none match our patterns
                    other_files = os.listdir(images_dir)
                    logger.error(f"Images directory contains {len(other_files)} files, but none are recognized image formats")
                    logger.info(f"First few files: {other_files[:5]}")
            
        return image_paths
    
    def _create_directories(self):
        """Create the necessary directory structure"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(self.output_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
                os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
                
            logger.info(f"Created directory structure in {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise
    
    def _process_split(self, image_paths, split, labels_dir, augment, num_augmentations):
        """
        Process a set of images for a specific split (train/val/test)
        
        Args:
            image_paths: List of image paths
            split: Split name ('train', 'val', 'test')
            labels_dir: Directory containing label files
            augment: Whether to apply augmentation
            num_augmentations: Number of augmented versions to create
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing {split} set ({len(image_paths)} images)...")
        
        stats = {
            'processed': 0,
            'skipped': 0,
            'augmented': 0,
            'missing_labels': 0,
            'errors': []
        }
        
        for img_path in tqdm(image_paths):
            try:
                # Get base name and corresponding label path
                img_filename = os.path.basename(img_path)
                base_name = os.path.splitext(img_filename)[0]
                label_path = os.path.join(labels_dir, f"{base_name}.txt")
                
                # Check if label exists
                if not os.path.exists(label_path):
                    logger.warning(f"No label found for {img_path}. Skipping.")
                    stats['missing_labels'] += 1
                    continue
                
                # Copy original image and label
                dest_img_path = os.path.join(self.output_dir, split, 'images', img_filename)
                dest_label_path = os.path.join(self.output_dir, split, 'labels', f"{base_name}.txt")
                
                shutil.copy(img_path, dest_img_path)
                shutil.copy(label_path, dest_label_path)
                
                stats['processed'] += 1
                
                # Apply augmentation for training set if requested
                if augment and split == 'train' and self.transformer:
                    self._augment_image(img_path, label_path, base_name, split, num_augmentations, stats)
            except Exception as e:
                error_msg = f"Error processing {img_path}: {str(e)}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
                stats['skipped'] += 1
        
        # Log summary
        logger.info(f"{split} set processing complete: {stats['processed']} processed, {stats['augmented']} augmented, {stats['missing_labels']} missing labels, {stats['skipped']} skipped")
        
        return stats
    
    def _augment_image(self, img_path, label_path, base_name, split, num_augmentations, stats):
        """Augment an image and save the results"""
        try:
            # Load original image
            img = Image.open(img_path)
            
            # Generate augmented versions
            for aug_idx in range(num_augmentations):
                # Augment
                augmented = self.transformer(img)
                
                # Save augmented image
                aug_name = f"{base_name}_aug{aug_idx}"
                aug_img_path = os.path.join(self.output_dir, split, 'images', f"{aug_name}.png")
                augmented.save(aug_img_path)
                
                # Copy label (same label for augmented version)
                aug_label_path = os.path.join(self.output_dir, split, 'labels', f"{aug_name}.txt")
                shutil.copy(label_path, aug_label_path)
                
                stats['augmented'] += 1
        except Exception as e:
            logger.error(f"Error augmenting {img_path}: {str(e)}")
            stats['skipped'] += 1
    
    def _create_dataset_yaml(self):
        """
        Create a YAML configuration file for YOLO training
        
        Returns:
            Path to the generated YAML file
        """
        try:
            # Try to load class names from classes.txt
            class_names = self._load_class_names()
            
            # Get absolute path to output directory
            abs_output_dir = os.path.abspath(self.output_dir)
            
            # Automatically detect the best path format based on the environment
            use_relative_paths = self._should_use_relative_paths()
            
            # Create YAML data with appropriate paths
            data = {
                'path': abs_output_dir,  # Base path is always absolute
            }
            
            # Set paths based on detection
            if use_relative_paths:
                # Use relative paths (works in most environments)
                data.update({
                    'train': './train/images',  # Relative paths from the base path
                    'val': './val/images',
                    'test': './test/images',
                })
                logger.info("Using relative paths in dataset YAML")
            else:
                # Use absolute paths (needed in some environments)
                data.update({
                    'train': os.path.join(abs_output_dir, 'train', 'images'),
                    'val': os.path.join(abs_output_dir, 'val', 'images'),
                    'test': os.path.join(abs_output_dir, 'test', 'images'),
                })
                logger.info("Using absolute paths in dataset YAML")
            
            # Add class information
            data.update({
                'nc': len(class_names),
                'names': class_names
            })
            
            # Write YAML file
            yaml_path = os.path.join(self.output_dir, 'dataset.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            # Verify the YAML file
            self._verify_yaml_paths(yaml_path)
            
            logger.info(f"Created dataset YAML file: {yaml_path}")
            return yaml_path
        except Exception as e:
            logger.error(f"Error creating dataset YAML: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_yaml()
    
    def _should_use_relative_paths(self):
        """Determine if relative paths should be used based on environment"""
        # Default to relative paths
        use_relative_paths = True
        
        # Check if we're in a special environment that requires absolute paths
        if 'COLAB_GPU' in os.environ or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            use_relative_paths = False
            logger.info("Detected cloud environment, using absolute paths")
            
        # Check for other indicators that might suggest using absolute paths
        if self.system_info['os'] == 'Windows':
            # Windows paths can sometimes cause issues with relative paths
            logger.info("Detected Windows OS, still using relative paths but with caution")
            
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.info("Detected virtual environment")
            
        return use_relative_paths
    
    def _load_class_names(self):
        """Load class names from classes.txt or use defaults"""
        # Try to load class names from classes.txt
        classes_file = os.path.join(os.path.dirname(self.output_dir), 'classes.txt')
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(class_names)} classes from {classes_file}")
                return class_names
            except Exception as e:
                logger.error(f"Error loading classes from {classes_file}: {str(e)}")
                
        # Try alternative locations
        alt_locations = [
            os.path.join(self.input_dir, 'classes.txt'),
            os.path.join(os.path.dirname(self.input_dir), 'classes.txt'),
            'classes.txt'
        ]
        
        for location in alt_locations:
            if os.path.exists(location):
                try:
                    with open(location, 'r') as f:
                        class_names = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(class_names)} classes from {location}")
                    return class_names
                except Exception as e:
                    logger.error(f"Error loading classes from {location}: {str(e)}")
        
        # Default class names
        logger.warning("Could not find classes.txt, using default alphanumeric classes")
        return list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    
    def _verify_yaml_paths(self, yaml_path):
        """
        Verify that the paths in the YAML file exist
        
        Args:
            yaml_path: Path to the YAML file
        """
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check if the base path exists
            if 'path' in data and not os.path.exists(data['path']):
                logger.warning(f"Base path '{data['path']}' does not exist")
            
            # Check if the train/val/test paths exist
            for split in ['train', 'val', 'test']:
                if split in data:
                    # Handle both relative and absolute paths
                    if data[split].startswith('./'):
                        # Relative path
                        full_path = os.path.join(data['path'], data[split][2:])
                    else:
                        # Absolute path
                        full_path = data[split]
                    
                    if not os.path.exists(full_path):
                        logger.warning(f"{split} path '{full_path}' does not exist")
                        
            logger.info("YAML path verification complete")
        except Exception as e:
            logger.error(f"Error verifying YAML paths: {str(e)}")
    
    def _create_fallback_yaml(self):
        """
        Create a fallback YAML file when the normal creation process fails
        
        Returns:
            Path to the generated fallback YAML file
        """
        try:
            # Get absolute path to output directory
            abs_output_dir = os.path.abspath(self.output_dir)
            
            # Create minimal YAML data with both relative and absolute paths
            data = {
                'path': abs_output_dir,
                'train': './train/images',  # Try relative path first
                'train_abs': os.path.join(abs_output_dir, 'train', 'images'),  # Fallback absolute path
                'val': './val/images',
                'val_abs': os.path.join(abs_output_dir, 'val', 'images'),
                'test': './test/images',
                'test_abs': os.path.join(abs_output_dir, 'test', 'images'),
                'nc': 62,  # Default to alphanumeric
                'names': list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            }
            
            # Write YAML file
            yaml_path = os.path.join(self.output_dir, 'dataset_fallback.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            logger.info(f"Created fallback dataset YAML file: {yaml_path}")
            
            # Create a README file explaining the fallback
            readme_path = os.path.join(self.output_dir, 'FALLBACK_README.txt')
            with open(readme_path, 'w') as f:
                f.write("FALLBACK DATASET CONFIGURATION\n")
                f.write("==============================\n\n")
                f.write("This is a fallback configuration created because the normal dataset preparation process encountered errors.\n\n")
                f.write("To use this configuration:\n")
                f.write("1. Open the dataset_fallback.yaml file\n")
                f.write("2. If training fails with the default paths, try the following:\n")
                f.write("   a. Remove the 'train', 'val', and 'test' entries\n")
                f.write("   b. Rename 'train_abs' to 'train', 'val_abs' to 'val', and 'test_abs' to 'test'\n\n")
                f.write("Error information:\n")
                f.write("- Check the console output for detailed error messages\n")
                f.write("- Make sure your data is organized correctly with images in 'images' directory and labels in 'labels' directory\n")
                f.write("- Verify that label files have the same base names as their corresponding images\n")
            
            return yaml_path
        except Exception as e:
            logger.error(f"Error creating fallback YAML: {str(e)}")
            raise
    
    def convert_to_yolo_format(self, input_dir, output_dir, image_width, image_height, class_mapping=None):
        """
        Convert annotations to YOLO format
        
        Args:
            input_dir: Directory containing original annotations
            output_dir: Directory to save YOLO format annotations
            image_width: Width of the images
            image_height: Height of the images
            class_mapping: Dictionary mapping original class names to YOLO class indices
            
        Returns:
            Number of converted annotations
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Default class mapping if not provided
            if not class_mapping:
                class_mapping = self._get_class_mapping()
            
            # Find all annotation files
            annotation_files = glob.glob(os.path.join(input_dir, '*.txt'))
            
            if not annotation_files:
                logger.warning(f"No annotation files found in {input_dir}")
                return 0
            
            converted = 0
            errors = 0
            
            for ann_file in tqdm(annotation_files):
                try:
                    # Read original annotation
                    with open(ann_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Convert to YOLO format
                    yolo_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class_name x_min y_min x_max y_max
                            class_name = parts[0]
                            x_min, y_min, x_max, y_max = map(float, parts[1:5])
                            
                            # Convert to YOLO format (normalized center coordinates and dimensions)
                            x_center = (x_min + x_max) / 2 / image_width
                            y_center = (y_min + y_max) / 2 / image_height
                            width = (x_max - x_min) / image_width
                            height = (y_max - y_min) / image_height
                            
                            # Validate values
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                                logger.warning(f"Invalid normalized coordinates in {ann_file}: {x_center}, {y_center}, {width}, {height}")
                                # Clip values to valid range
                                x_center = max(0, min(x_center, 1))
                                y_center = max(0, min(y_center, 1))
                                width = max(0, min(width, 1))
                                height = max(0, min(height, 1))
                            
                            # Get class index
                            class_idx = class_mapping.get(class_name, 0)
                            
                            # Create YOLO format line
                            yolo_line = f"{class_idx} {x_center} {y_center} {width} {height}\n"
                            yolo_lines.append(yolo_line)
                    
                    # Write YOLO format annotation
                    output_file = os.path.join(output_dir, os.path.basename(ann_file))
                    with open(output_file, 'w') as f:
                        f.writelines(yolo_lines)
                    
                    converted += 1
                except Exception as e:
                    logger.error(f"Error converting {ann_file}: {str(e)}")
                    errors += 1
            
            logger.info(f"Converted {converted} annotations, {errors} errors")
            return converted
        except Exception as e:
            logger.error(f"Error in convert_to_yolo_format: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    def _get_class_mapping(self):
        """Get class mapping from classes.txt or use defaults"""
        # Try to load from classes.txt
        classes_file = os.path.join(os.path.dirname(self.output_dir), 'classes.txt')
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r') as f:
                    classes = [line.strip() for line in f if line.strip()]
                class_mapping = {cls: idx for idx, cls in enumerate(classes)}
                logger.info(f"Loaded class mapping for {len(classes)} classes from {classes_file}")
                return class_mapping
            except Exception as e:
                logger.error(f"Error loading class mapping from {classes_file}: {str(e)}")
        
        # Try alternative locations
        alt_locations = [
            os.path.join(self.input_dir, 'classes.txt'),
            os.path.join(os.path.dirname(self.input_dir), 'classes.txt'),
            'classes.txt'
        ]
        
        for location in alt_locations:
            if os.path.exists(location):
                try:
                    with open(location, 'r') as f:
                        classes = [line.strip() for line in f if line.strip()]
                    class_mapping = {cls: idx for idx, cls in enumerate(classes)}
                    logger.info(f"Loaded class mapping for {len(classes)} classes from {location}")
                    return class_mapping
                except Exception as e:
                    logger.error(f"Error loading class mapping from {location}: {str(e)}")
        
        # Create default mapping for alphanumeric characters
        logger.warning("Could not find classes.txt, using default alphanumeric class mapping")
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        return {char: idx for idx, char in enumerate(chars)}
