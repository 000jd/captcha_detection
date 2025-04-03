import os
import sys
import yaml
import torch
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.captcha_transform import CAPTCHATransform
from src.utils.captcha_preprocessor import CAPTCHAPreprocessor
from src.utils.multi_captcha_detector import MultiCAPTCHADetector
from src.utils.dataset_manager import DatasetManager

def train_model(data_yaml, 
                model_size='n',
                epochs=100, 
                batch_size=16, 
                img_size=640, 
                save_dir='runs/train',
                pretrained_weights=None):
    """
    Train a YOLO model on CAPTCHA dataset
    
    Args:
        data_yaml: Path to the YAML file with dataset configuration
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
        save_dir: Directory to save training results
        pretrained_weights: Path to pretrained weights (optional)
        
    Returns:
        Path to the best trained model
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml}")
        
    # Initialize a new YOLO model
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"Loading pretrained weights from {pretrained_weights}")
        model = YOLO(pretrained_weights)
    else:
        print(f"Initializing new YOLO11{model_size} model")
        model = YOLO(f'yolo11{model_size}.pt')
    
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
        print(f"Best model saved to {best_model_path}")
        return best_model_path
    else:
        print(f"Warning: Best model not found at {best_model_path}")
        return None

def test_model(model_path, data_yaml, img_size=640, batch_size=16, save_dir='runs/val'):
    """
    Test a trained YOLO model on CAPTCHA dataset
    
    Args:
        model_path: Path to the trained model
        data_yaml: Path to the YAML file with dataset configuration
        img_size: Image size for testing
        batch_size: Batch size
        save_dir: Directory to save testing results
        
    Returns:
        Dictionary with test results
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml}")
        
    # Load model
    model = YOLO(model_path)
    
    # Test the model
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        project=os.path.dirname(save_dir),
        name=os.path.basename(save_dir)
    )
    
    return results

def prepare_training_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, augment=True, num_augmentations=5):
    """
    Prepare training data for CAPTCHA detection
    
    Args:
        input_dir: Directory containing original images and labels
        output_dir: Directory to save the processed dataset
        train_ratio: Ratio of images for training set
        val_ratio: Ratio of images for validation set
        test_ratio: Ratio of images for test set
        augment: Whether to apply augmentation
        num_augmentations: Number of augmented versions to create per original image
        
    Returns:
        Path to the generated YAML file
    """
    # Initialize dataset manager
    dataset_manager = DatasetManager(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Prepare dataset
    stats = dataset_manager.prepare_dataset(
        augment=augment,
        num_augmentations=num_augmentations
    )
    
    print(f"Dataset preparation complete:")
    print(f"  - Total original images: {stats['total_images']}")
    print(f"  - Training images: {stats['train']['processed']}")
    print(f"  - Validation images: {stats['val']['processed']}")
    print(f"  - Test images: {stats['test']['processed']}")
    print(f"  - Augmented images: {stats['train']['augmented']}")
    print(f"  - Missing labels: {stats['train']['missing_labels'] + stats['val']['missing_labels'] + stats['test']['missing_labels']}")
    
    return stats['yaml_path']

def visualize_dataset(data_yaml, num_samples=5):
    """
    Visualize samples from the dataset
    
    Args:
        data_yaml: Path to the YAML file with dataset configuration
        num_samples: Number of samples to visualize
        
    Returns:
        None
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml}")
        
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get paths
    train_path = data_config['train']
    val_path = data_config['val']
    class_names = data_config['names']
    
    # Visualize training samples
    _visualize_split_samples(train_path, 'train', class_names, num_samples)
    
    # Visualize validation samples
    _visualize_split_samples(val_path, 'val', class_names, num_samples)

def _visualize_split_samples(images_dir, split_name, class_names, num_samples):
    """
    Visualize samples from a specific split
    
    Args:
        images_dir: Directory containing images
        split_name: Name of the split ('train', 'val', 'test')
        class_names: List of class names
        num_samples: Number of samples to visualize
        
    Returns:
        None
    """
    import glob
    import random
    import numpy as np
    
    # Get all image paths
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg')) + \
                  glob.glob(os.path.join(images_dir, '*.jpeg')) + \
                  glob.glob(os.path.join(images_dir, '*.png'))
    
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
    
    # Randomly select samples
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    # Create figure
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    if len(samples) == 1:
        axes = [axes]
    
    # Plot each sample
    for i, img_path in enumerate(samples):
        # Load image
        img = Image.open(img_path)
        
        # Get corresponding label path
        label_path = os.path.join(
            os.path.dirname(os.path.dirname(img_path)),
            'labels',
            os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        )
        
        # Plot image
        axes[i].imshow(img)
        axes[i].set_title(f"{split_name} sample {i+1}")
        axes[i].axis('off')
        
        # Plot bounding boxes if label exists
        if os.path.exists(label_path):
            # Load labels
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Get image dimensions
            width, height = img.size
            
            # Plot each bounding box
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_idx = int(parts[0])
                    x_center, y_center, box_width, box_height = map(float, parts[1:5])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_min = int((x_center - box_width/2) * width)
                    y_min = int((y_center - box_height/2) * height)
                    x_max = int((x_center + box_width/2) * width)
                    y_max = int((y_center + box_height/2) * height)
                    
                    # Plot bounding box
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         fill=False, edgecolor='red', linewidth=2)
                    axes[i].add_patch(rect)
                    
                    # Add class label
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                    axes[i].text(x_min, y_min - 5, class_name, color='red',
                                fontsize=10, backgroundcolor='white')
    
    plt.tight_layout()
    plt.savefig(f"{split_name}_samples.png")
    plt.close()
    
    print(f"Saved {split_name} samples visualization to {split_name}_samples.png")


