import numpy as np
import cv2 as cv
import pandas as pd
import os
import glob
import scipy.spatial.transform as tf

def point_grid(frame, step):
    height, widht = frame.shape[:2]
    step = 30
    points = []
    for y in range(0, height, step):
        for x in range(0, widht, step):
            points.append([x, y])
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

def depth_from_flow(pixel_coord, pixel_flow, eigen_vel, eigen_rot_rate):
    if np.linalg.norm(eigen_vel) < 0.1:
        return None
    x, y = pixel_coord
    A = np.array([
        [-1, 0, x],
        [0, -1, y],
    ])
    B = np.array([
        [x*y, -(1 + x**2), y],
        [(1 + y**2), -x*y, -x],
    ])
    trans_flow = pixel_flow - B @ eigen_rot_rate
    t = A @ eigen_vel
    Z = np.dot(t,t) / np.dot(trans_flow, t)
    return Z



def world_frame_vel_to_local(eigen_vel, eigen_att):
    R = tf.Rotation.from_euler('xyz', eigen_att).as_matrix()
    local_vel = R.T @ eigen_vel
    return local_vel

def world_frame_rot_to_local(eigen_rot_rate, eigen_att):
    R = tf.Rotation.from_euler('xyz', eigen_att).as_matrix()
    local_rot = R.T @ eigen_rot_rate
    return local_rot

# 1. Setup Paths
folder_path = "/home/ruben/Downloads/AE4317_2019_datasets/cyberzoo_poles/20190121-135009"
csv_path = os.path.join(folder_path, "/home/ruben/Downloads/AE4317_2019_datasets/cyberzoo_poles/20190121-135121.csv") # Update filename if different

# 2. Load and Prepare Data
df = pd.read_csv(csv_path)
image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

# Parameters for Flow
feature_params = dict(maxCorners=50, qualityLevel=0.000001, minDistance=20, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 3. Initialize
old_frame = cv.imread(image_files[0])
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

w, h = old_gray.shape[::-1]

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
    v = np.array([drone_state['vel_x'], drone_state['vel_y'], drone_state['vel_z']])
    r = np.array([drone_state['rate_p'], drone_state['rate_q'], drone_state['rate_r']])
    att = np.array([drone_state['att_phi'], drone_state['att_theta'], drone_state['att_psi']])
    v = world_frame_vel_to_local(v, att)
    r = world_frame_rot_to_local(r, att)
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
            dy = b - d
            u = np.array([dx / w, dy / h]) # measured flow in pixels
            p = np.array([a/ w, b / h]) # pixel coordinate
            z = depth_from_flow(p, u, v, r)

            color = (0, 200-int(z*50), 200+int(z*50)) if z is not None else (255, 255, 255)
            cv.circle(frame, (int(a), int(b)), 4, color, -1)

        p0 = good_new.reshape(-1, 1, 2)

    # Display Telemetry on screen
    # cv.putText(frame, f"Vel_X: {v_x:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv.putText(frame, f"Yaw_R: {yaw_rate:.2f} rad/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv.imshow('Sync Flow', frame)
    if cv.waitKey(50) & 0xff == 27: break
    old_gray = frame_gray.copy()

cv.destroyAllWindows()