import cv2
import os
import glob
import re
import numpy as np

# ==========================================================
# FINAL SETTINGS (Hardcoded from your Tuning)
# ==========================================================

""" 
# Real
CONF = {
    "GATE_W": 1.0,
    "GATE_H": 1.0,
    "H_MIN": 2,
    "H_MAX": 17,
    "S_MIN": 134,
    "V_MIN": 126,
    "WARP": 0.013,        # From your "Shape: Warp Tolerance"
    "CONF_RATIO": 0.17,   # Your 13% confidence
    "ALPHA": 0.2,         # Smoothing factor (0.1 = smooth, 0.9 = twitchy)
    "MIN_AREA": 700,
    "LINE_THICK": 8
}
"""

# Simulation
CONF = {
    "GATE_W": 1.0,
    "GATE_H": 1.0,
    "H_MIN": 18,           # Shifted up: Cuts out red/orange, starts at Golden Yellow
    "H_MAX": 32,           # Shifted down: Tightens the range around pure Yellow
    "S_MIN": 165,          # Increased: Requires more "pure" color (less washed out)
    "V_MIN": 130,          # Increased: Ignores dark shadows and brown textures
    "WARP": 0.013,         
    "CONF_RATIO": 0.25,    # Slightly stricter: Requires more of the shape to match
    "ALPHA": 0.3,         
    "MIN_AREA": 450,       # Slightly increased: Filters out small distant noise
    "LINE_THICK": 6,
    "MAX_SOLIDITY": 0.60   # Keeps it focused on hollow gate shapes
}
OBJ_POINTS = np.array([
    [-CONF["GATE_W"]/2,  CONF["GATE_H"]/2, 0], 
    [ CONF["GATE_W"]/2,  CONF["GATE_H"]/2, 0], 
    [ CONF["GATE_W"]/2, -CONF["GATE_H"]/2, 0], 
    [-CONF["GATE_W"]/2, -CONF["GATE_H"]/2, 0]  
], dtype=np.float32)

def get_full_dataset(folder_path):
    extensions = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(folder_path, ext)))
    all_files.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', f)])
    return all_files

def solve_gate_spatial(approx, img_w, img_h):
    focal_length = img_w # Focal step 1.0
    cam_matrix = np.array([[focal_length, 0, img_w/2], [0, focal_length, img_h/2], [0, 0, 1]], dtype=np.float32)
    
    # Sort corners for PnP
    pts = approx.reshape(4, 2).astype(np.float32)
    pts = pts[np.argsort(pts[:, 1]), :]
    top_pts = pts[:2, :][np.argsort(pts[:2, 0]), :]
    bot_pts = pts[2:, :][np.argsort(pts[2:, 0])[::-1], :]
    ordered_pts = np.vstack([top_pts, bot_pts])

    success, rvec, tvec = cv2.solvePnP(OBJ_POINTS, ordered_pts, cam_matrix, np.zeros((4,1)))
    if success:
        dist = np.linalg.norm(tvec)
        ox = np.mean(ordered_pts[:, 0]) - (img_w / 2)
        oy = np.mean(ordered_pts[:, 1]) - (img_h / 2)
        # Yaw calculation relative to normal
        yaw = np.degrees(np.arctan2(tvec[0], tvec[2]))
        return {"off_x": ox, "off_y": oy, "dist": dist, "yaw": yaw, "rvec": rvec, "tvec": tvec, "cam": cam_matrix}
    return None

def gate_generator(image_list):
    """ Yields telemetry and handles visualization. """
    # Smoothing states
    s_dist, s_ox, s_oy, s_yaw = 0, 0, 0, 0
    alpha = CONF["ALPHA"]

    for img_path in image_list:
        frame = cv2.imread(img_path)
        if frame is None: continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = frame.shape[:2]

        # 1. Image Processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([CONF["H_MIN"], CONF["S_MIN"], CONF["V_MIN"]]), 
                                np.array([CONF["H_MAX"], 255, 255]))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 40, 120)
        
        # Dilate to help with distant thin gates
        dilated = cv2.dilate(edged, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        telemetry = None

        for cnt in contours:
            if cv2.contourArea(cnt) < CONF["MIN_AREA"]: continue
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, CONF["WARP"] * peri, True)

            if len(approx) == 4:
                # Confidence check
                l_mask = np.zeros(gray.shape, dtype="uint8")
                cv2.drawContours(l_mask, [cnt], -1, 255, thickness=CONF["LINE_THICK"])
                ratio = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=l_mask)) / (cv2.countNonZero(l_mask) + 1e-5)

                if ratio > CONF["CONF_RATIO"]:
                    data = solve_gate_spatial(approx, w, h)
                    if data:
                        # Apply Alpha Smoothing
                        s_dist = (alpha * data['dist']) + (1 - alpha) * s_dist
                        s_ox = (alpha * data['off_x']) + (1 - alpha) * s_ox
                        s_oy = (alpha * data['off_y']) + (1 - alpha) * s_oy
                        s_yaw = (alpha * data['yaw']) + (1 - alpha) * s_yaw
                        
                        telemetry = {
                            "dist": round(s_dist, 3),
                            "off_x": int(s_ox),
                            "off_y": int(s_oy),
                            "yaw": round(float(s_yaw), 2)
                        }
                        
                        # --- DRAW 3D AXIS & CONTOUR ---
                        # Axis points: Origin, X(red), Y(green), Z(blue)
                        axis_pts = np.array([[0,0,0], [0.4,0,0], [0,0.4,0], [0,0,0.4]], dtype=np.float32)
                        img_pts, _ = cv2.projectPoints(axis_pts, data['rvec'], data['tvec'], data['cam'], np.zeros((4,1)))
                        img_pts = img_pts.astype(int)
                        origin = tuple(img_pts[0].ravel())
                        
                        cv2.line(frame, origin, tuple(img_pts[1].ravel()), (0, 0, 255), 2) # X
                        cv2.line(frame, origin, tuple(img_pts[2].ravel()), (0, 255, 0), 2) # Y
                        cv2.line(frame, origin, tuple(img_pts[3].ravel()), (255, 0, 0), 4) # Z (Normal)
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                        break 

        cv2.imshow('Live Stream Debug', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        yield telemetry

    cv2.destroyAllWindows()

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    # --- CHOOSE YOUR DATASET PATH ---
    dataset_path = "Data/AE4317_2019_datasets/sim_poles_panels_mats/20190121-161931"
    # dataset_path = "Data/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935"
    # dataset_path = "Data/AE4317_2019_datasets/cyberzoo_aggressive_flight/20190121-144646"

    images = get_full_dataset(dataset_path)

    print(f"{'STATUS':<12} | {'DIST':<8} | {'OFF_X':<8} | {'OFF_Y':<8} | {'YAW'}")
    print("-" * 65)

    for data in gate_generator(images):
        if data:
            print(f"{'TRACKING':<12} | {data['dist']:<8.2f} | {data['off_x']:<8} | {data['off_y']:<8} | {data['yaw']:>5.1f}°")
        else:
            print(f"{'LOST':<12} | {'--':<8} | {'--':<8} | {'--':<8} | {'--'}", end='\r')