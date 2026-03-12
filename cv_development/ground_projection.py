"""
ground_projection.py

Estimate obstacle distance from camera using bounding box + pixel count.
"""
import os
import glob
from dataclasses import dataclass
import math
from orange_pole_detector import OrangePoleDetector
import cv2
import numpy as np
import sys
sys.path.append("/home/roan2003/paparazzi")
from pathfinding_testing.image_correction import K, D, Knew


def undistort_bbox_box(box, K, D, Knew):
    corners = np.array([
        [box.x, box.y],
        [box.x + box.w, box.y],
        [box.x + box.w, box.y + box.h],
        [box.x, box.y + box.h],
    ], dtype=np.float32).reshape(-1, 1, 2)

    undistorted = cv2.fisheye.undistortPoints(corners, K, D, P=Knew)
    undistorted = undistorted.reshape(-1, 2)

    xs = undistorted[:, 0]
    ys = undistorted[:, 1]
    x_min, y_min = float(xs.min()), float(ys.min())
    x_max, y_max = float(xs.max()), float(ys.max())

    return (x_min, y_min, x_max - x_min, y_max - y_min), undistorted

def OrangeBox_to_distance(x_min,w,Knew, real_width=0.35):
    # Assume the box is on the ground and we know its real-world width
    Orange_FoV = w/(2*(Knew[0,2]))*np.pi # width in normalized image coords
    distance = real_width/(2*np.sin(Orange_FoV/2))*np.sin(np.pi/2 - Orange_FoV/2) # distance from camera to box
    heading= (x_min + w/2 - Knew[0,2])/(2*Knew[0,2])*np.pi# heading in radians)
    return distance, heading

# Configuration
INPUT_FOLDER = '/home/roan2003/paparazzi/cv_development/20190121-160844'
# Create an output list or file to save data
results_log = []

detector = OrangePoleDetector()

# Get all image paths
image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))

for path in image_paths:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        continue
        
    rot_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    result = detector.detect(rot_bgr)
    
    # 1. Generate the base drawing (bounding boxes)
    draw = detector.draw(rot_bgr, result)
    
    if result['boxes']:
        box = result['boxes'][0]
        undistorted_box, _ = undistort_bbox_box(box, K, D, Knew)
        
        distance, heading = OrangeBox_to_distance(
            undistorted_box[0], 
            undistorted_box[2], 
            Knew
        )
        
        # 2. Add the calculated text to the 'draw' image
        text_dist = f"Dist: {distance:.2f}m"
        text_head = f"Head: {math.degrees(heading):.1f}deg"
        
        # Draw Distance
        cv2.putText(draw, text_dist, (box.x, box.y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw Heading
        cv2.putText(draw, text_head, (box.x, box.y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        print(f"Img: {os.path.basename(path)} | {text_dist} | {text_head}")
        
    # 3. Show the image (now containing both the box AND the text)
    cv2.imshow("Batch Processing", draw)
    
    # Wait for a key press: 'q' to quit, any other key for next image
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()