# scripts/split_dataset.py
import os
import shutil
import random
from pathlib import Path

def split_dataset(image_dir, output_dir, train_ratio=0.8, val_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        image_dir: Directory containing all images
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    """
    
    # Create directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in os.listdir(image_dir) 
              if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # Copy files
    def copy_files(image_list, split_name):
        for img_name in image_list:
            # Copy image
            src_img = os.path.join(image_dir, img_name)
            dst_img = os.path.join(output_dir, 'images', split_name, img_name)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label if exists
            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_label = os.path.join(image_dir, 'labels', label_name)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_dir, 'labels', split_name, label_name)
                shutil.copy2(src_label, dst_label)
    
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')
    
    print(f"Dataset split completed:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Create dataset YAML file
    create_dataset_yaml(output_dir)

def create_dataset_yaml(output_dir):
    """Create YOLO dataset configuration file"""
    
    # Define vehicle classes based on your images
    vehicle_classes = {
        0: 'car',
        1: 'bus',
        2: 'truck',
        3: 'motorcycle',
        4: 'van',
        5: 'pickup',
        6: 'trailer',
        7: 'ambulance',
        8: 'traveller'  # Based on your images showing "Traveller"
    }
    
    yaml_content = f"""# Traffic Monitoring Dataset Configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(vehicle_classes)}

# Class names
names: {list(vehicle_classes.values())}
"""
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset YAML created: {yaml_path}")
    
    # Also save classes.txt
    classes_path = os.path.join('config', 'classes.txt')
    with open(classes_path, 'w') as f:
        for class_name in vehicle_classes.values():
            f.write(f"{class_name}\n")
    
    print(f"Classes file created: {classes_path}")

if __name__ == "__main__":
    # Example usage
    image_directory = "data/raw_images"  # Folder containing your 100+ images
    output_directory = "data/dataset"
    
    split_dataset(image_directory, output_directory)