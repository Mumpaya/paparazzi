"""
Clean YOLO Dataset
Remove images and labels that weren't in your annotated poles_gt.csv
"""

import os
import pandas as pd

# Configuration
CSV_FILE = '/home/roan2003/paparazzi/cv_development/poles_gt.csv'
YOLO_IMAGES_DIR = '/home/roan2003/paparazzi/cv_development/yolo_dataset/images/train'
YOLO_LABELS_DIR = '/home/roan2003/paparazzi/cv_development/yolo_dataset/labels/train'

def get_annotated_images():
    """Get list of images that have actual box coordinates in poles_gt.csv"""
    df = pd.read_csv(CSV_FILE)
    
    annotated = []
    for idx, row in df.iterrows():
        filename = row['filename']
        box1 = row['box_1_coords']
        box2 = row['box_2_coords']
        
        # Check if either box has coordinates
        box1_has = not pd.isna(box1) and str(box1).strip() and str(box1) != 'nan'
        box2_has = not pd.isna(box2) and str(box2).strip() and str(box2) != 'nan'
        
        if box1_has or box2_has:
            annotated.append(filename)
    
    return set(annotated)


def clean_yolo_dataset():
    """Remove unnannotated images from YOLO dataset"""
    annotated_images = get_annotated_images()
    
    print("CLEANING YOLO DATASET")
    print("=" * 60)
    print(f"Annotated images from poles_gt.csv: {len(annotated_images)}\n")
    
    # Get all current files
    current_images = set([f for f in os.listdir(YOLO_IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    current_labels = set([f for f in os.listdir(YOLO_LABELS_DIR) if f.endswith('.txt')])
    
    # Find files to remove
    images_to_remove = current_images - annotated_images
    labels_to_remove = set()
    
    for img in images_to_remove:
        label = img.replace('.jpg', '.txt').replace('.png', '.txt')
        labels_to_remove.add(label)
    
    print(f"Images to REMOVE: {len(images_to_remove)}")
    print(f"Labels to REMOVE: {len(labels_to_remove)}\n")
    
    if len(images_to_remove) > 0:
        print("Files to remove:")
        for img in sorted(images_to_remove):
            print(f"  - {img}")
        print()
    
    # Confirm before deletion
    response = input("Proceed with deletion? (yes/no): ").strip().lower()
    
    if response == 'yes':
        # Remove image files
        for img in images_to_remove:
            img_path = os.path.join(YOLO_IMAGES_DIR, img)
            os.remove(img_path)
            print(f"Deleted: {img}")
        
        # Remove label files
        for label in labels_to_remove:
            label_path = os.path.join(YOLO_LABELS_DIR, label)
            if os.path.exists(label_path):
                os.remove(label_path)
                print(f"Deleted: {label}")
        
        print(f"\n{'='*60}")
        print(f"✓ Cleanup complete!")
        print(f"Remaining images: {len(current_images - images_to_remove)}")
        print(f"{'='*60}\n")
    else:
        print("Cleanup cancelled.")


if __name__ == "__main__":
    clean_yolo_dataset()
