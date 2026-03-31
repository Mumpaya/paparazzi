"""
YOLO Predictions vs Ground Truth Visualization
Compares model predictions with ground truth annotations for visual validation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Configuration
DATASET_DIR = '/home/roan2003/paparazzi/cv_development/yolo_dataset'
# Use your trained model (change to yolov8n.pt if you want the base model)
MODEL_PATH = '/home/roan2003/paparazzi/runs/detect/train2/weights/best.pt'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images', 'train')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels', 'train')

# Class names (from dataset.yaml)
CLASS_NAMES = {0: 'pole', 1: 'gate'}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}  # BGR format

# Confidence threshold for predictions
# Lower threshold needed for small datasets with poor training
# Model trained on only 26 images with very low confidence predictions
CONFIDENCE_THRESHOLD = 0.05  # Show predictions >= 5% confidence


def load_ground_truth(label_file):
    """
    Load ground truth bounding boxes from YOLO label file
    Format: class_id x_center y_center width height (normalized 0-1)
    
    Returns: List of (class_id, x_center, y_center, width, height)
    """
    ground_truth = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                ground_truth.append(values)
    return ground_truth


def get_images_with_objects(min_objects=1):
    """
    Get list of images that have ground truth annotations
    
    Args:
        min_objects: Minimum number of objects required
    
    Returns:
        List of image filenames with annotations
    """
    images_with_objects = []
    
    for label_file in os.listdir(LABELS_DIR):
        if not label_file.endswith('.txt'):
            continue
        
        label_path = os.path.join(LABELS_DIR, label_file)
        object_count = 0
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            object_count = len([l for l in lines if l.strip()])
        
        if object_count >= min_objects:
            image_name = label_file.replace('.txt', '.jpg')
            images_with_objects.append(image_name)
    
    return sorted(images_with_objects)


def normalize_to_pixel(bbox_normalized, img_width, img_height):
    """
    Convert normalized YOLO bbox to pixel coordinates
    Returns: (x1, y1, x2, y2) - top-left and bottom-right corners
    """
    class_id, x_center, y_center, width, height = bbox_normalized
    
    # Convert from normalized to pixel coordinates
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    
    return int(class_id), x1, y1, x2, y2


def draw_bbox(image, bbox, label, color, line_thickness=2):
    """
    Draw bounding box on image
    bbox: (x1, y1, x2, y2) in pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
    
    # Draw label text
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def visualize_single_image(image_name, model, show_plot=False, save_output=True, output_dir=None):
    """
    Visualize predictions vs ground truth for a single image
    
    Args:
        image_name: Name of the image file
        model: YOLO model instance
        show_plot: Whether to display plot (set to False for headless environments)
        save_output: Whether to save visualization to file
        output_dir: Directory to save output images
    """
    image_path = os.path.join(IMAGES_DIR, image_name)
    label_path = os.path.join(LABELS_DIR, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Create output directory if needed
    if save_output and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare image
    original_image = cv2.imread(image_path)
    img_height, img_width = original_image.shape[:2]
    
    # Create copies for visualization
    img_with_gt = original_image.copy()
    img_with_pred = original_image.copy()
    img_combined = original_image.copy()
    
    # Load ground truth
    ground_truth = load_ground_truth(label_path)
    
    # Draw ground truth
    for gt in ground_truth:
        class_id, x1, y1, x2, y2 = normalize_to_pixel(gt, img_width, img_height)
        label = f"{CLASS_NAMES[class_id]} (GT)"
        color = CLASS_COLORS[class_id]
        cv2.rectangle(img_with_gt, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img_combined, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img_with_gt, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img_with_gt, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(img_combined, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img_combined, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print(f"\nProcessing: {image_name}")
    results = model.predict(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    predictions = []
    if results and len(results) > 0:
        detections = results[0]
        if hasattr(detections, 'boxes'):
            for box in detections.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                predictions.append((class_id, x1, y1, x2, y2, confidence))
                
                # Draw on prediction image
                label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
                color = CLASS_COLORS[class_id]
                draw_bbox(img_with_pred, (x1, y1, x2, y2), label, color, 2)
                
                # Draw on combined image with bright cyan color and thick lines
                label_combined = f"{CLASS_NAMES[class_id]}: {confidence:.2f} (PRED)"
                # Use bright CYAN (0, 255, 255) for predictions - very distinct from green/red
                cv2.rectangle(img_combined, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
                # Draw label with background
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(label_combined, font, 0.6, 2)[0]
                cv2.rectangle(
                    img_combined,
                    (x1, y1 - text_size[1] - 5),
                    (x1 + text_size[0], y1),
                    (0, 255, 255),
                    -1
                )
                cv2.putText(
                    img_combined,
                    label_combined,
                    (x1, y1 - 5),
                    font,
                    0.6,
                    (0, 0, 0),  # Black text on cyan background
                    2
                )
    
    print(f"Ground Truth: {len(ground_truth)} objects")
    print(f"Predictions: {len(predictions)} objects")
    
    if len(predictions) > 0:
        print("\nPrediction Details:")
        for i, (class_id, x1, y1, x2, y2, conf) in enumerate(predictions, 1):
            print(f"  {i}. {CLASS_NAMES[class_id]} - Confidence: {conf:.3f}")
    
    # Display/Save results
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convert BGR to RGB for display
        axes[0].imshow(cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Ground Truth Annotations', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img_with_pred, cv2.COLOR_BGR2RGB))
        axes[1].set_title('YOLO Predictions', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Ground Truth (Green/Red) + Predictions (Orange)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    if save_output and output_dir:
        base_name = os.path.splitext(image_name)[0]
        
        # Save individual images
        gt_path = os.path.join(output_dir, f"{base_name}_01_ground_truth.jpg")
        pred_path = os.path.join(output_dir, f"{base_name}_02_predictions.jpg")
        combined_path = os.path.join(output_dir, f"{base_name}_03_combined.jpg")
        
        cv2.imwrite(gt_path, img_with_gt)
        cv2.imwrite(pred_path, img_with_pred)
        cv2.imwrite(combined_path, img_combined)
        
        print(f"\nSaved visualizations to:")
        print(f"  - {gt_path}")
        print(f"  - {pred_path}")
        print(f"  - {combined_path}")
    
    return {
        'image': img_combined,
        'ground_truth': ground_truth,
        'predictions': predictions
    }


def batch_visualize(num_samples=None, save_output=True, use_annotated_only=True):
    """
    Visualize multiple images from the dataset
    
    Args:
        num_samples: Number of images to process (None = all images)
        save_output: Whether to save output images
        use_annotated_only: If True, only process images with ground truth annotations
    """
    # Create output directory
    output_dir = '/home/roan2003/paparazzi/cv_development/yolo_results'
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Get list of images
    if use_annotated_only:
        print("Finding images with ground truth annotations...")
        image_files = get_images_with_objects(min_objects=1)
        print(f"Found {len(image_files)} images with annotations")
    else:
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
    
    if len(image_files) == 0:
        print(f"No suitable images found")
        return
    
    # Determine number of samples to process
    if num_samples is None:
        sample_images = image_files  # Process all images
    else:
        sample_images = image_files[:num_samples]
    
    for idx, image_name in enumerate(sample_images, 1):
        print(f"\n{'='*60}")
        print(f"Processing {idx}/{len(sample_images)}: {image_name}")
        print(f"{'='*60}")
        visualize_single_image(image_name, model, show_plot=False, 
                             save_output=save_output, output_dir=output_dir)
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"Total images processed: {len(sample_images)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    """
    Main execution: Visualize YOLO predictions vs ground truth
    
    USAGE EXAMPLES:
    ===============
    
    1. Visualize ALL images with annotations (default):
       python yolo_predictions_vs_ground_truth.py
    
    2. Visualize only first N images:
       From Python: batch_visualize(num_samples=5, save_output=True)
    
    3. Visualize a specific image:
       From Python: visualize_single_image('100011279.jpg', YOLO(MODEL_PATH), save_output=True, output_dir='yolo_results')
    
    4. Visualize images with objects (interactive approach):
       From Python:
       >>> from yolo_predictions_vs_ground_truth import *
       >>> model = YOLO(MODEL_PATH)
       >>> images = get_images_with_objects()
       >>> visualize_single_image(images[0], model, save_output=True, output_dir='yolo_results')
    """
    print("\n" + "="*60)
    print("YOLO PREDICTIONS VS GROUND TRUTH VISUALIZATION")
    print("="*60)
    
    # Visualize ALL images with annotations from the dataset
    # Set num_samples to limit (e.g., num_samples=5 for first 5 images only)
    batch_visualize(num_samples=None, save_output=True, use_annotated_only=True)
