import os
import yaml
import shutil

class ClassManager:
    """
    Manages CAPTCHA character classes for object detection models.
    Handles loading, organizing, and generating configuration files for classes.
    """
    
    def __init__(self, classes_file=None):
        """
        Initialize the class manager
        
        Args:
            classes_file: Path to the file containing class names
        """
        self.classes_file = classes_file
        self.classes = self._load_classes()
        
    def _load_classes(self):
        """Load class names from file"""
        if not self.classes_file or not os.path.exists(self.classes_file):
            # Default to basic alphanumeric characters if no file provided
            return list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        
        with open(self.classes_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def get_class_id(self, class_name):
        """Get the ID for a class name"""
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1
    
    def get_class_name(self, class_id):
        """Get the name for a class ID"""
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return None
    
    def get_all_classes(self):
        """Get all class names"""
        return self.classes
    
    def get_num_classes(self):
        """Get the number of classes"""
        return len(self.classes)
    
    def generate_data_yaml(self, output_path, train_path, val_path, test_path=None):
        """
        Generate a YAML configuration file for YOLO training
        
        Args:
            output_path: Path to save the YAML file
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data (optional)
            
        Returns:
            Path to the generated YAML file
        """
        data = {
            'path': os.path.dirname(os.path.abspath(output_path)),
            'train': train_path,
            'val': val_path,
            'nc': len(self.classes),
            'names': self.classes
        }
        
        if test_path:
            data['test'] = test_path
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write YAML file
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
        return output_path
    
    def distribute_classes(self, output_dir):
        """
        Distribute class names to appropriate files in the project structure
        
        Args:
            output_dir: Base directory for the project
            
        Returns:
            Dictionary with paths to generated files
        """
        files = {}
        
        # Create classes directory
        classes_dir = os.path.join(output_dir, 'data', 'classes')
        os.makedirs(classes_dir, exist_ok=True)
        
        # Write classes.txt to the classes directory
        classes_path = os.path.join(classes_dir, 'classes.txt')
        with open(classes_path, 'w') as f:
            f.write('\n'.join(self.classes))
        files['classes_txt'] = classes_path
        
        # Copy original classes file if it exists
        if self.classes_file and os.path.exists(self.classes_file):
            shutil.copy(self.classes_file, classes_dir)
        
        # Generate YAML configuration
        data_yaml_path = os.path.join(output_dir, 'data', 'captcha_data.yaml')
        self.generate_data_yaml(
            data_yaml_path,
            train_path='./train/images',
            val_path='./val/images',
            test_path='./test/images'
        )
        files['data_yaml'] = data_yaml_path
        
        # Generate class mapping file (JSON)
        import json
        class_map = {i: cls for i, cls in enumerate(self.classes)}
        class_map_path = os.path.join(classes_dir, 'class_map.json')
        with open(class_map_path, 'w') as f:
            json.dump(class_map, f, indent=2)
        files['class_map'] = class_map_path
        
        return files
    
    def create_class_examples(self, output_dir, font_size=24, font_path=None):
        """
        Create example images for each class
        
        Args:
            output_dir: Directory to save example images
            font_size: Font size for the examples
            font_path: Path to a font file (optional)
            
        Returns:
            Dictionary mapping class names to image paths
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            os.makedirs(output_dir, exist_ok=True)
            examples = {}
            
            # Try to load font
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                # Use default font
                font = ImageFont.load_default()
                
            # Create example for each class
            for cls in self.classes:
                # Skip empty classes
                if not cls:
                    continue
                    
                # Create image
                img = Image.new('RGB', (32, 32), color=(255, 255, 255))
                draw = ImageDraw.Draw(img)
                
                # Calculate text position to center it
                text_width, text_height = draw.textsize(cls, font=font)
                position = ((32 - text_width) // 2, (32 - text_height) // 2)
                
                # Draw text
                draw.text(position, cls, fill=(0, 0, 0), font=font)
                
                # Save image
                img_path = os.path.join(output_dir, f"class_{cls}.png")
                img.save(img_path)
                examples[cls] = img_path
                
            return examples
        except Exception as e:
            print(f"Error creating class examples: {e}")
            return {}
