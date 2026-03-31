"""
Create Annotated Dataset
Generates and saves annotated images with ground truth bounding boxes drawn on them
Useful for visual verification of annotations before model training
"""

import os
import cv2
import shutil
from pathlib import Path

# Configuration
DATASET_DIR = '/home/roan2003/paparazzi/cv_development/yolo_dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images', 'train')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels', 'train')
OUTPUT_DIR = '/home/roan2003/paparazzi/cv_development/annotated_images'

# Class names and colors (BGR format)
CLASS_NAMES = {0: 'pole', 1: 'gate'}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}  # Green for poles, Red for gates

def load_ground_truth(label_file):
    """
    Load ground truth bounding boxes from YOLO label file
    Format: class_id x_center y_center width height (normalized 0-1)
    """
    ground_truth = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    values = list(map(float, line.strip().split()))
                    ground_truth.append(values)
    return ground_truth


def normalize_to_pixel(bbox_normalized, img_width, img_height):
    """
    Convert normalized YOLO bbox to pixel coordinates
    Returns: (class_id, x1, y1, x2, y2)
    """
    class_id, x_center, y_center, width, height = bbox_normalized
    
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    
    # Clamp to image boundaries
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    return int(class_id), x1, y1, x2, y2


def draw_annotations(image, ground_truth_list):
    """
    Draw all ground truth annotations on the image
    
    Args:
        image: Input image
        ground_truth_list: List of normalized bounding boxes
    
    Returns:
        Annotated image
    """
    img_height, img_width = image.shape[:2]
    annotated_image = image.copy()
    
    for gt in ground_truth_list:
        class_id, x1, y1, x2, y2 = normalize_to_pixel(gt, img_width, img_height)
        color = CLASS_COLORS[class_id]
        class_name = CLASS_NAMES[class_id]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 5
        
        # Draw background rectangle for text
        cv2.rectangle(
            annotated_image,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return annotated_image


def create_annotated_dataset():
    """
    Create annotated versions of all dataset images with ground truth boxes drawn
    Saves to OUTPUT_DIR organized in subdirectories
    """
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with_objects_dir = os.path.join(OUTPUT_DIR, 'with_objects')
    empty_dir = os.path.join(OUTPUT_DIR, 'empty')
    
    os.makedirs(with_objects_dir, exist_ok=True)
    os.makedirs(empty_dir, exist_ok=True)
    
    print(f"Creating annotated dataset...")
    print(f"Images source: {IMAGES_DIR}")
    print(f"Labels source: {LABELS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    
    print(f"Found {len(image_files)} images\n")
    
    stats = {
        'total': 0,
        'with_objects': 0,
        'empty': 0,
        'total_objects': 0
    }
    
    for idx, image_name in enumerate(image_files, 1):
        image_path = os.path.join(IMAGES_DIR, image_name)
        label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(LABELS_DIR, label_name)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_name}")
            continue
        
        # Load ground truth
        ground_truth = load_ground_truth(label_path)
        
        # Draw annotations
        annotated_image = draw_annotations(image, ground_truth)
        
        # Determine output directory based on whether image has objects
        if len(ground_truth) > 0:
            output_path = os.path.join(with_objects_dir, image_name)
            stats['with_objects'] += 1
            stats['total_objects'] += len(ground_truth)
        else:
            output_path = os.path.join(empty_dir, image_name)
            stats['empty'] += 1
        
        # Save annotated image
        cv2.imwrite(output_path, annotated_image)
        stats['total'] += 1
        
        # Print progress
        if idx % 20 == 0 or idx == len(image_files):
            print(f"Processed {idx}/{len(image_files)} images...")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ANNOTATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total']}")
    print(f"Images with objects: {stats['with_objects']}")
    print(f"Empty images: {stats['empty']}")
    print(f"Total objects annotated: {stats['total_objects']}")
    print(f"\nOutput locations:")
    print(f"  With objects: {with_objects_dir}")
    print(f"  Empty: {empty_dir}")
    print(f"{'='*60}\n")


def create_sample_comparison(num_samples=5):
    """
    Create side-by-side comparison images of original vs annotated
    Useful for quick visual verification
    """
    comparison_dir = os.path.join(OUTPUT_DIR, 'comparison_samples')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get images with objects
    with_objects_dir = os.path.join(OUTPUT_DIR, 'with_objects')
    image_files = sorted([f for f in os.listdir(with_objects_dir) if f.endswith(('.jpg', '.png'))])[:num_samples]
    
    print(f"Creating comparison samples ({len(image_files)} samples)...\n")
    
    for image_name in image_files:
        # Load original and annotated
        original_path = os.path.join(IMAGES_DIR, image_name)
        annotated_path = os.path.join(with_objects_dir, image_name)
        
        original = cv2.imread(original_path)
        annotated = cv2.imread(annotated_path)
        
        if original is None or annotated is None:
            continue
        
        # Create side-by-side comparison
        h, w = original.shape[:2]
        comparison = cv2.hconcat([original, annotated])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Annotated", (w + 10, 30), font, 1, (0, 255, 0), 2)
        
        # Save
        output_name = f"comparison_{image_name}"
        output_path = os.path.join(comparison_dir, output_name)
        cv2.imwrite(output_path, comparison)
        
        print(f"Saved: {output_name}")
    
    print(f"\nComparison samples saved to: {comparison_dir}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CREATING ANNOTATED DATASET")
    print("="*60 + "\n")
    
    # Create full annotated dataset
    create_annotated_dataset()
    
    # Create comparison samples
    create_sample_comparison(num_samples=5)
    
    print("✓ Done! Your annotated images are ready for review.")
