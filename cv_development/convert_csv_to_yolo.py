"""
Convert poles_gt.csv to YOLO Format
Converts pixel coordinates from poles_gt.csv to normalized YOLO format (0-1 range)
"""

import os
import json
import pandas as pd
import cv2

# Configuration
CSV_FILE = '/home/roan2003/paparazzi/cv_development/poles_gt.csv'
IMAGES_DIR = '/home/roan2003/paparazzi/cv_development/yolo_dataset/images/train'
LABELS_DIR = '/home/roan2003/paparazzi/cv_development/yolo_dataset/labels/train'

def pixel_to_normalized(x1, y1, x2, y2, img_width, img_height):
    """
    Convert pixel coordinates to normalized YOLO format
    Input: x1, y1, x2, y2 (pixel coordinates)
    Output: class_id, x_center, y_center, width, height (normalized 0-1)
    """
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Clamp values to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def convert_csv_to_yolo():
    """Convert all annotations from poles_gt.csv to YOLO format"""
    df = pd.read_csv(CSV_FILE)
    
    print("CONVERTING poles_gt.csv TO YOLO FORMAT")
    print("=" * 60)
    
    converted = 0
    skipped = 0
    objects_count = 0
    
    for idx, row in df.iterrows():
        filename = row['filename']
        box1_str = row['box_1_coords']
        box2_str = row['box_2_coords']
        
        # Check if image has annotations
        box1_has = not pd.isna(box1_str) and str(box1_str).strip() and str(box1_str) != 'nan'
        box2_has = not pd.isna(box2_str) and str(box2_str).strip() and str(box2_str) != 'nan'
        
        if not (box1_has or box2_has):
            skipped += 1
            continue
        
        # Load image to get dimensions
        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {filename}")
            skipped += 1
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image: {filename}")
            skipped += 1
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Parse boxes and convert to YOLO format
        yolo_lines = []
        
        if box1_has:
            try:
                box1 = json.loads(box1_str)
                x1, y1, x2, y2 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
                x_center, y_center, width, height = pixel_to_normalized(x1, y1, x2, y2, img_width, img_height)
                # Class 0 = pole (from your class mapping)
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                objects_count += 1
            except Exception as e:
                print(f"Error parsing box1 in {filename}: {e}")
        
        if box2_has:
            try:
                box2 = json.loads(box2_str)
                x1, y1, x2, y2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
                x_center, y_center, width, height = pixel_to_normalized(x1, y1, x2, y2, img_width, img_height)
                # Class 0 = pole
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                objects_count += 1
            except Exception as e:
                print(f"Error parsing box2 in {filename}: {e}")
        
        # Write to YOLO label file
        label_name = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(LABELS_DIR, label_name)
        
        with open(label_path, 'w') as f:
            f.writelines(yolo_lines)
        
        converted += 1
        
        # Progress indicator
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(df)} images...")
    
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Images converted: {converted}")
    print(f"Images skipped (no annotations): {skipped}")
    print(f"Total objects annotated: {objects_count}")
    print(f"Output directory: {LABELS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    convert_csv_to_yolo()
