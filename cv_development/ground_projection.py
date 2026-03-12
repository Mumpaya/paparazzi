"""
ground_projection.py

Estimate obstacle distance from camera using bounding box + pixel count.
"""

from dataclasses import dataclass
import math
from pathfinding_testing import image_correction

# use functions
img = image_correction.load_image('/some/path.jpg')
image_correction.run_calibration()

import numpy as np
import cv2

def undistort_bbox_box(box, K, D, Knew):
    # box: (x,y,w,h) in distorted image pixel coords
    corners = np.array([
        [box[0], box[1]],
        [box[0] + box[2], box[1]],
        [box[0] + box[2], box[1] + box[3]],
        [box[0], box[1] + box[3]],
    ], dtype=np.float32).reshape(-1,1,2)

    # undistort to new camera matrix
    undistorted = cv2.fisheye.undistortPoints(corners, K, D, P=Knew)
    undistorted = undistorted.reshape(-1,2)

    x_coords = undistorted[:,0]
    y_coords = undistorted[:,1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    w = x_max - x_min
    h = y_max - y_min
    return (x_min, y_min, w, h), undistorted