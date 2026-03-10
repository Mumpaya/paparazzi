import numpy as np
import image_correction as ic

import os
import glob
import numpy as np
import cv2 as cv
import pandas as pd

# --- 1. SETUP AND PATHS ---
folder_path = "/home/roan2003/Documents/Optical flow tryout/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
csv_path = "/home/roan2003/Documents/Optical flow tryout/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"

# Optional: use script folder and folder relative paths
# script_dir = os.path.dirname(os.path.abspath(__file__))
# folder_path = os.path.join(script_dir, "front_cam_gate/20260306-104712")
# csv_path = os.path.join(script_dir, "your_file.csv")

print("folder_path=", folder_path)
print("csv_path=", csv_path)

df = pd.read_csv(csv_path)
image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) +
                     glob.glob(os.path.join(folder_path, "*.png")) +
                     glob.glob(os.path.join(folder_path, "*.jpeg")))

if len(image_files) < 2:
    raise ValueError(f"Need at least two images to compute optical flow, found {len(image_files)}")

print(f"Loaded {len(image_files)} image files")

# first image (apply image correction/undistortion)
frame1 = ic.load_image(image_files[0])
if frame1 is None:
    raise RuntimeError(f"Unable to read or correct {image_files[0]}")

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

for i in range(1, len(image_files)):
    frame2 = ic.load_image(image_files[i])
    if frame2 is None:
        print(f"Skipping missing/bad frame: {image_files[i]}")
        continue

    nxt = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, nxt, None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow("opticalflow", bgr)
    key = cv.waitKey(100) & 0xFF  # slower display: 100 ms per frame
    if key == 27:
        break
    elif key == ord("s"):
        cv.imwrite("opticalfb.png", frame2)
        cv.imwrite("opticalhsv.png", bgr)

    prvs = nxt

cv.destroyAllWindows()
