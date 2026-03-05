import numpy as np
import cv2 as cv
import pandas as pd
import os
import glob

# --- 1. SETUP AND PATHS ---
folder_path = "/home/roan2003/Documents/Optical flow tryout/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
csv_path = os.path.join(folder_path, "/home/roan2003/Documents/Optical flow tryout/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv")

df = pd.read_csv(csv_path)
image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

# --- 2. PARAMETERS ---
MAG_THRESHOLD = 2.5    
MAX_MAG_CAP = 15.0     

# Distance Logic
DISTANCE_THRESHOLD = 10  
V_X_MIN = 0.09
V_Y_MIN = 0.09
V_Z_MIN = 0.09 
YAW_RATE_THRESHOLD = 0.3            
SMOOTHING_WINDOW = 2       

# FOVEATION PARAMETERS (Focus on the center 200x200 pixels)
FOCUS_SIZE = 150 

intensity_history = []
old_frame = cv.imread(image_files[0])
old_frame = cv.rotate(old_frame, cv.ROTATE_90_COUNTERCLOCKWISE)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255 

for i in range(1, len(image_files)):
    frame = cv.imread(image_files[i])
    if frame is None: break
    
    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # --- SYNC LOGIC ---
    filename = os.path.basename(image_files[i])
    img_time = float(filename.replace(".jpg", "")) / 1000000.0 
    idx = (df['time'] - img_time).abs().idxmin()
    drone_state = df.iloc[idx]
    yaw_rate = drone_state['rate_r']
    v_x = drone_state['vel_x']
    v_y = drone_state['vel_y']
    v_z = drone_state['vel_z']

    # --- DENSE OPTICAL FLOW ---
    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 
                                       pyr_scale=0.5, levels=3, winsize=15, 
                                       iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # --- EGO-MOTION COMPENSATION ---
    f_constant = 50.0 
    flow[..., 1] += (yaw_rate * f_constant) 

    # --- INTENSITY FILTERING ---
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mag_filtered = np.where(mag < MAG_THRESHOLD, 0, mag)
    mag_filtered = np.where(mag_filtered > MAX_MAG_CAP, MAX_MAG_CAP, mag_filtered)

    # --- REFINED CENTER FOCUS (FOVEATION) ---
    h, w = frame.shape[:2]
    
    # Calculate start/end indices for the center focus area
    start_x = (w // 2) - (FOCUS_SIZE // 2)
    end_x   = (w // 2) + (FOCUS_SIZE // 2)
    start_y = (h // 2) - (FOCUS_SIZE // 2)
    end_y   = (h // 2) + (FOCUS_SIZE // 2)
    
    # Extract only the central pixels
    center_roi = mag_filtered[start_y:end_y, start_x:end_x]
    current_intensity = np.mean(center_roi)

    # Distance Metric Calculation
    distance_metric = abs(v_x) / (current_intensity + 1e-6)

    intensity_history.append(distance_metric)
    if len(intensity_history) > SMOOTHING_WINDOW:
        intensity_history.pop(0)
    avg_dist_metric = np.mean(intensity_history)

    # --- VISUALIZATION ---
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag_filtered, None, 0, 255, cv.NORM_MINMAX)
    bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Draw the specific Focus Box
    cv.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    
    if avg_dist_metric < DISTANCE_THRESHOLD and abs(v_x) > V_X_MIN and abs(v_y) > V_Y_MIN and abs(v_z) > V_Z_MIN and abs(yaw_rate) < YAW_RATE_THRESHOLD:
        cv.putText(frame, "OBSTACLE IN PATH", (20, 100), 
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

    cv.putText(frame, f"Dist Metric: {avg_dist_metric:.4f}", (20, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(frame, f"Velocity X: {abs(v_x):.4f}", (20, 70), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(frame, f"Velocity Y: {abs(v_y):.4f}", (20, 100), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(frame, f"Velocity Z: {abs(v_z):.4f}", (20, 130), 
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    

    cv.imshow('Dense Flow', bgr_flow)
    cv.imshow('Drone Focus View', frame)
    
    if cv.waitKey(30) & 0xff == 27: break
    old_gray = frame_gray.copy()

cv.destroyAllWindows()