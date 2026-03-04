import numpy as np
import cv2 as cv
import pandas as pd
import os
import glob

# 1. Setup Paths
folder_path = "/home/roan2003/Documents/Optical flow tryout/AE4317_2019_datasets/cyberzoo_poles/20190121-135009"
csv_path = os.path.join(folder_path, "/home/roan2003/Documents/Optical flow tryout/AE4317_2019_datasets/cyberzoo_poles/20190121-135121.csv") # Update filename if different

# 2. Load and Prepare Data
df = pd.read_csv(csv_path)
image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

# Parameters for Flow
feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=5, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 3. Initialize
old_frame = cv.imread(image_files[0])
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

for i in range(1, len(image_files)):
    frame = cv.imread(image_files[i])
    if frame is None: break
    
    # --- SYNC LOGIC ---
    # Extract timestamp from filename (75044792 -> 75.044792)
    filename = os.path.basename(image_files[i])
    img_time = float(filename.replace(".jpg", "")) / 1000000.0 # Convert to same scale as CSV
    
    # Find the closest row in CSV
    idx = (df['time'] - img_time).abs().idxmin()
    drone_state = df.iloc[idx]
    
    # Extract useful telemetry
    v_x = drone_state['vel_x'] # Forward velocity
    yaw_rate = drone_state['rate_r'] # Turning speed
    # ------------------

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Refresh points if needed
    if p0 is None or len(p0) < 20:
        p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # Calculate Flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            dx = a - c
            
            # --- EGO-MOTION COMPENSATION ---
            # If the drone yaws, pixels move. We "subtract" the turn to see the OBSTACLE.
            # 'f' is a scaling factor for the camera focal length/resolution
            f = 50.0 
            true_dx = dx + (yaw_rate * f) 

            # Color points based on TRUE motion, not just image motion
            color = (0, 0, 255) if abs(true_dx) > 10 else (0, 255, 0)
            cv.circle(frame, (int(a), int(b)), 4, color, -1)

        p0 = good_new.reshape(-1, 1, 2)

    # Display Telemetry on screen
    cv.putText(frame, f"Vel_X: {v_x:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(frame, f"Yaw_R: {yaw_rate:.2f} rad/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv.imshow('Sync Flow', frame)
    if cv.waitKey(50) & 0xff == 27: break
    old_gray = frame_gray.copy()

cv.destroyAllWindows()