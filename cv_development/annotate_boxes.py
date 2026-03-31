import cv2
import os
import csv
import json
import pandas as pd
import random

# Configuration
IMG_DIR = '/home/roan2003/paparazzi/cv_development/cyberzoo_poles/20190121-135009'
CSV_FILE = '/home/roan2003/paparazzi/cv_development/poles_gt.csv'

class BoundingBoxAnnotator:
    def __init__(self, img_dir, csv_file):
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.image = None
        self.original_image = None
        self.current_img_name = None
        
        # Load existing CSV
        if os.path.isfile(csv_file):
            self.df = pd.read_csv(csv_file)
        else:
            print(f"Warning: {csv_file} not found. Creating new file.")
            self.df = pd.DataFrame(columns=['filename'])

    def has_box_coordinates(self, img_name):
        """Check if an image already has box coordinates in the CSV"""
        if img_name not in self.df['filename'].values:
            return False
        
        # Check if any box columns exist for this image
        row = self.df[self.df['filename'] == img_name].iloc[0]
        box_cols = [col for col in self.df.columns if col.startswith('box_') and col.endswith('_coords')]
        
        for col in box_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                return True
        
        return False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = {'x1': x, 'y1': y, 'x2': x, 'y2': y}
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_box:
                self.current_box['x2'] = x
                self.current_box['y2'] = y
                # Show preview of box while dragging
                self.image = self.original_image.copy()
                self.draw_boxes()
                self.draw_current_box()
                cv2.imshow('Annotate Boxes', self.image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_box and (self.current_box['x1'] != self.current_box['x2'] or 
                                     self.current_box['y1'] != self.current_box['y2']):
                self.boxes.append(self.current_box)
                self.image = self.original_image.copy()
                self.draw_boxes()
                cv2.imshow('Annotate Boxes', self.image)
            self.current_box = None

    def draw_boxes(self):
        """Draw all saved boxes on image"""
        for i, box in enumerate(self.boxes):
            x1, y1 = box['x1'], box['y1']
            x2, y2 = box['x2'], box['y2']
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.image, f'Box {i+1}', (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_current_box(self):
        """Draw the box currently being drawn"""
        if self.current_box:
            x1, y1 = self.current_box['x1'], self.current_box['y1']
            x2, y2 = self.current_box['x2'], self.current_box['y2']
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 165, 255), 2)

    def annotate_image(self, img_name):
        """Annotate a single image"""
        img_path = os.path.join(self.img_dir, img_name)
        
        if not os.path.isfile(img_path):
            print(f"Image not found: {img_path}")
            return False
        
        self.current_img_name = img_name
        self.original_image = cv2.imread(img_path)
        self.image = self.original_image.copy()
        self.boxes = []
        
        try:
            if cv2.getWindowProperty('Annotate Boxes', cv2.WND_PROP_VISIBLE) < 0:
                cv2.namedWindow('Annotate Boxes')
        except:
            cv2.namedWindow('Annotate Boxes')
        
        cv2.setMouseCallback('Annotate Boxes', self.mouse_callback)
        cv2.imshow('Annotate Boxes', self.image)
        
        print(f"\n=== Annotating: {img_name} ===")
        print("Instructions:")
        print("  - Click and drag to draw bounding boxes")
        print("  - 's' to save and move to next image")
        print("  - 'u' to undo last box")
        print("  - 'c' to clear all boxes")
        print("  - 'q' to quit")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if chr(key) == 's':
                print(f"Saved {len(self.boxes)} box(es) for {img_name}")
                return True
                
            elif chr(key) == 'u':
                if self.boxes:
                    self.boxes.pop()
                    self.image = self.original_image.copy()
                    self.draw_boxes()
                    cv2.imshow('Annotate Boxes', self.image)
                    print(f"Undid last box. Remaining: {len(self.boxes)}")
                    
            elif chr(key) == 'c':
                self.boxes = []
                self.image = self.original_image.copy()
                cv2.imshow('Annotate Boxes', self.image)
                print("Cleared all boxes")
                
            elif chr(key) == 'q':
                print("Quitting without saving")
                cv2.destroyAllWindows()
                return None

    def save_to_csv(self, img_name, boxes):
        """Save boxes for an image to CSV by merging with existing data"""
        # Find the row with this image
        if img_name in self.df['filename'].values:
            idx = self.df[self.df['filename'] == img_name].index[0]
        else:
            # Add new row if image not in CSV
            idx = len(self.df)
            self.df.loc[idx, 'filename'] = img_name
        
        # Add box data columns
        if len(boxes) > 0:
            # Create columns for each box
            for i, box in enumerate(boxes):
                col_name = f'box_{i+1}_coords'
                self.df.loc[idx, col_name] = json.dumps(box)
            
            print(f"Added {len(boxes)} box(es) to {img_name}")
        
        # Save updated CSV
        self.df.to_csv(self.csv_file, index=False)
        print(f"Saved to {self.csv_file}")

    def run(self):
        """Main annotation loop"""
        # Get list of images
        img_files = sorted([f for f in os.listdir(self.img_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not img_files:
            print(f"No images found in {self.img_dir}")
            return
        
        print(f"Found {len(img_files)} total images")
        
        # Filter out images that already have box coordinates
        remaining_images = [img for img in img_files if not self.has_box_coordinates(img)]
        skipped_images = len(img_files) - len(remaining_images)
        
        print(f"Already annotated: {skipped_images} images")
        print(f"Remaining to annotate: {len(remaining_images)} images")
        
        if len(remaining_images) == 0:
            print("All images have been annotated!")
            return
        
        # Randomize the order of remaining images
        random.shuffle(remaining_images)
        print(f"\nRandomizing image order...\n")
        
        for idx, img_name in enumerate(remaining_images):
            print(f"\n[{idx+1}/{len(remaining_images)}] Processing {img_name}")
            
            result = self.annotate_image(img_name)
            
            if result is True:  # Saved
                self.save_to_csv(img_name, self.boxes)
            elif result is None:  # Quit
                break
        
        cv2.destroyAllWindows()
        print(f"\nAnnotations saved to {self.csv_file}")


if __name__ == "__main__":
    annotator = BoundingBoxAnnotator(IMG_DIR, CSV_FILE)
    annotator.run()
