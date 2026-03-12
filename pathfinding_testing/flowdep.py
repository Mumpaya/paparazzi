import numpy as np
import image_correction as ic

import os
import glob
import numpy as np
import cv2 as cv
import pandas as pd
import scipy.spatial.transform as tf

def world_frame_vel_to_local(eigen_vel, eigen_att):
    R = tf.Rotation.from_euler('xyz', eigen_att).as_matrix()
    local_vel = R.T @ eigen_vel
    R2 = tf.Rotation.from_euler('zyx', np.array([0.0, np.pi/2, -np.pi/2])).as_matrix()
    return R2.T @ local_vel

def world_frame_rot_to_local(eigen_rot_rate, eigen_att):
    R = tf.Rotation.from_euler('xyz', eigen_att).as_matrix()
    local_rot = R.T @ eigen_rot_rate
    R2 = tf.Rotation.from_euler('zyx', np.array([0.0, np.pi/2, -np.pi/2])).as_matrix()
    return R2.T @ local_rot

def flow_to_bgr(flow):
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / 10 *255, 0, 255).astype(np.uint8)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

# --- 1. SETUP AND PATHS ---
# folder_path = "/home/roan2003/Documents/Optical flow tryout/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
# csv_path = "/home/roan2003/Documents/Optical flow tryout/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
folder_path = r"/home/ruben/Downloads/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935"
csv_path = r"/home/ruben/Downloads/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142943.csv"

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
start = 100
frame1 = ic.load_image(image_files[start])
if frame1 is None:
    raise RuntimeError(f"Unable to read or correct {image_files[0]}")

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
prvs_t = ic.get_img_time_from_filename(image_files[0])

for i in range(start+1, len(image_files), 1):
    frame2 = ic.load_image(image_files[i])
    if frame2 is None:
        print(f"Skipping missing/bad frame: {image_files[i]}")
        continue
    nxt_t = ic.get_img_time_from_filename(image_files[i])

    dt = nxt_t- prvs_t

    # Find the closest row in CSV
    drone_state_prvs = df.iloc[(df['time'] - prvs_t).abs().idxmin()]
    drone_state_nxt = df.iloc[(df['time'] - nxt_t).abs().idxmin()]
    att = np.array([drone_state_nxt['att_phi'], drone_state_nxt['att_theta'], drone_state_nxt['att_psi']])
    att0 = np.array([drone_state_prvs['att_phi'], drone_state_prvs['att_theta'], drone_state_prvs['att_psi']])
    v = np.array([drone_state_nxt['vel_x'], drone_state_nxt['vel_y'], drone_state_nxt['vel_z']])
    r = (np.array([drone_state_nxt['rate_p'], drone_state_nxt['rate_q'], drone_state_nxt['rate_r']]) + np.array([drone_state_prvs['rate_p'], drone_state_prvs['rate_q'], drone_state_prvs['rate_r']]) )/ 2
    dr = (att - att0)
    v = world_frame_vel_to_local(v, att)
    r = world_frame_rot_to_local(r, att)
    dr = world_frame_rot_to_local(dr, att)
    print("r*dt=" , r*dt)
    print("dr=", dr)
    prvs_t = nxt_t

    nxt = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, nxt, None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    rot_flow = np.zeros_like(flow)
    for v in range(flow.shape[0]):
        for u in range(flow.shape[1]):
            x, y = (u - ic.cx) / ic.fx, (v - ic.cy) / ic.fy
            B = np.array([
                [x*y, -(1 + x**2), y],
                [(1 + y**2), -x*y, -x],
            ])
            ur = B @ (dr)
            ur *= np.array([ic.fx, ic.fy])
            rot_flow[v, u] = ur


    bgr = flow_to_bgr(flow - rot_flow)
    cv.imshow("opticalflow", bgr)
    cv.imshow("rotational_flow", flow_to_bgr(rot_flow))
    cv.imshow("base", prvs)

    key = cv.waitKey(1) & 0xFF  # slower display: 100 ms per frame
    if key == 27:
        break
    elif key == ord("s"):
        cv.imwrite("opticalfb.png", frame2)
        cv.imwrite("opticalhsv.png", bgr)

    prvs = nxt

cv.destroyAllWindows()
