from Denseflow_class import DenseOpticalFlowManager
from pole_detector import PoleDetector
import numpy as np
import cv2 as cv
import pandas as pd
import os
import glob

# --- 1. INITIALIZATION ---
# Detects poles based on vertical geometry (height > width)
pole_detector = PoleDetector(min_aspect_ratio=2.0)
# Manages dense flow, ego-motion, and ROI cropping
flow_manager = DenseOpticalFlowManager(mag_threshold=2.5)

# Tuning Parameters
DISTANCE_THRESHOLD = 10.0  
V_MIN = 0.05 
SMOOTHING_WINDOW = 5       
intensity_history = []

# Paths
folder_path = "/home/roan2003/paparazzi/cv_development/front_cam_gate-20260310T082155Z-1-001/front_cam_gate/20260306-104712"
csv_path = os.path.join(folder_path, "/home/roan2003/paparazzi/cv_development/front_cam_gate-20260310T082155Z-1-001/front_cam_gate/20260306-105523.csv")

# Load Telemetry
df = pd.read_csv(csv_path)
image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

# --- 2. PROCESSING LOOP ---
for i in range(len(image_files)):
    frame = cv.imread(image_files[i])
    if frame is None: break
    
    # Rotate frame 90° counterclockwise to upright (pole detector expects this)
    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    # --- SYNC TELEMETRY ---
    filename = os.path.basename(image_files[i])
    img_time = float(filename.replace(".jpg", "")) / 1000000.0
    idx = (df['time'] - img_time).abs().idxmin() 
    drone_state = df.iloc[idx]  
    
    v_x = drone_state['vel_x']
    yaw_rate = drone_state['rate_r']

    # --- STAGE 1: GEOMETRIC DETECTION ---
    # Finds vertical lines and clusters them into pole boxes
    detection_results = pole_detector.detect(frame)
    poles = detection_results["raw_boxes"]

    # --- STAGE 2: TEMPORAL OPTICAL FLOW ---
    if poles:
        # Select the pole box with the largest area (likely the closest/most dangerous)
        target_pole = max(poles, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        
        # Calculate Dense Flow ONLY inside the target_pole box
        # This provides: filtered magnitude, density map, and ROI intensity
        mag, density, heat, roi_val = flow_manager.process_frame(frame, yaw_rate, roi_box=target_pole)
        
        # Distance Metric: Velocity / Flow Expansion Intensity
        dist_metric = abs(v_x) / (roi_val + 1e-6)
    else:
        # If no pole is detected, we still process the frame to update 'old_gray' state
        # We pass a tiny dummy ROI or simply update internal state
        _ = flow_manager.process_frame(frame, yaw_rate, roi_box=[0,0,1,1])
        dist_metric = 100.0 
        heat = None
    
    # On first frame, process_frame returns None for heatmap; use blank instead
    if heat is None:
        heat = np.zeros_like(frame)

    # --- 3. DECISION LOGIC & SMOOTHING ---
    intensity_history.append(dist_metric)
    if len(intensity_history) > SMOOTHING_WINDOW:
        intensity_history.pop(0)
    avg_dist = np.mean(intensity_history)

    # --- 4. VISUALIZATION ---
    # Draw pole detection segments (green) and boxes (red)
    overlay_frame = pole_detector.draw(frame, detection_results)
    
    # Display distance metric and warnings
    cv.putText(overlay_frame, f"Dist Metric: {avg_dist:.4f}", (20, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if avg_dist < DISTANCE_THRESHOLD and abs(v_x) > V_MIN:
        cv.putText(overlay_frame, "CRITICAL DISTANCE!", (50, 150), 
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

    # Show the results (Full Landscape)
    cv.imshow('Integrated Tracker (Landscape)', overlay_frame)
    cv.imshow('Motion Heatmap (ROI)', heat)
    
    delay = 200  # ms
    if cv.waitKey(delay) & 0xff == 27: break
cv.destroyAllWindows()
