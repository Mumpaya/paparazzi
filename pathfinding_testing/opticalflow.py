import numpy as np
import cv2 as cv
import pandas as pd
import os
import glob
import scipy.spatial.transform as tf
import image_correction as ic

def point_grid(frame, step):
    height, widht = frame.shape[:2]
    step = 30
    points = []
    for y in range(0, height, step):
        for x in range(0, widht, step):
            points.append([x, y])
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

def depth_from_flow(pixel_coord, pixel_flow, eigen_vel, eigen_rot_rate,
                    vel_thresh=0.01, denom_thresh=0.005, t_thresh=0.01):
    """
    Estimate depth Z from normalised optical flow.

    Guards against instability:
      - ignore frames with negligible body translation
      - ignore points where the projected translation vector t is too small
        (the point happens to be near the Focus of Expansion)
      - ignore when denominator dot(u_trans, t) is near zero
      - clamp result to a physically plausible range [z_min, z_max]
    """
    if np.linalg.norm(eigen_vel) < vel_thresh:
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

    # Guard: projected translation at this pixel must be non-negligible
    if np.linalg.norm(t) < t_thresh:
        return None

    denom = np.dot(trans_flow, t)

    # Guard: denominator close to zero → Z undefined
    if abs(denom) < denom_thresh:
        return None

    Z = np.dot(t, t) / denom

    # Only physically meaningful, positive depth within a plausible range
    # if Z < z_min or Z > z_max:
    #     return None

    return Z

def world_frame_vel_to_local(eigen_vel, eigen_att):
    R = tf.Rotation.from_euler('xyz', eigen_att).as_matrix()
    local_vel = R.T @ eigen_vel
    return local_vel

def world_frame_rot_to_local(eigen_rot_rate, eigen_att):
    R = tf.Rotation.from_euler('xyz', eigen_att).as_matrix()
    local_rot = R.T @ eigen_rot_rate
    return local_rot


def normalize_pixel_coords(p):
    x = (p[0] - ic.cx) / ic.fx
    y = (p[1] - ic.cy) / ic.fy
    return np.array([x, y])

def normalize_pixel_flow(u):
    u_x = u[0] / ic.fx
    u_y = u[1] / ic.fy
    return np.array([u_x, u_y])


def get_img_time_from_filename(filename):
    return float(os.path.basename(filename).replace(".jpg", "")) / 1000000.0

def draw_point_with_depth(frame, pt, z, fac=100):
    """Draw a filled circle at pt and a small label with the z value to its right."""
    a, b = int(round(pt[0])), int(round(pt[1]))

    # color based on z (BGR for OpenCV), clamp to [0,255]
    if z is None:
        circle_color = (255, 255, 255)
        text = "n/a"
    else:
        delta = int(z * fac)
        blue = 0
        green = max(0, min(255, 200 - delta))
        red = max(0, min(255, 200 + delta))
        circle_color = (blue, green, red)
        text = f"{z:.2f} m"

    # draw circle
    cv.circle(frame, (a, b), 4, circle_color, -1)

    # prepare text background and text color (choose contrasting text color)
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    (tw, th), baseline = cv.getTextSize(text, font, scale, thickness)
    x0 = a + 6
    y0 = b - th // 2 - baseline
    x1 = x0 + tw
    y1 = b + th // 2

    # clamp rectangle coords to image bounds
    h_img, w_img = frame.shape[:2]
    x0 = max(0, min(w_img - 1, x0))
    x1 = max(0, min(w_img - 1, x1))
    y0 = max(0, min(h_img - 1, y0))
    y1 = max(0, min(h_img - 1, y1))

    cv.rectangle(frame, (x0, y0), (x1, y1), circle_color, cv.FILLED)

    # choose white or black text depending on brightness of the background
    brightness = (circle_color[0] + circle_color[1] + circle_color[2]) / 3
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

    cv.putText(frame, text, (x0, y1 - baseline), font, scale, text_color, thickness, cv.LINE_AA)


# 1. Setup Paths 20190121-142935
# folder_path = r"C:\Users\super\Downloads\AE4317_2019_datasets\AE4317_2019_datasets\cyberzoo_poles_panels_mats\20190121-142935"
# csv_path = r"C:\Users\super\Downloads\AE4317_2019_datasets\AE4317_2019_datasets\cyberzoo_poles_panels_mats\20190121-142943.csv"
# folder_path = r"C:\Users\super\Downloads\AE4317_2019_datasets\AE4317_2019_datasets\sim_poles_panels_mats\20190121-161931"
# csv_path = r"C:\Users\super\Downloads\AE4317_2019_datasets\AE4317_2019_datasets\sim_poles_panels_mats\20190121-161955.csv"

# folder_path = r"C:\Users\super\Downloads\own_datasets-20260306T115639Z-3-001\own_datasets\front_cam_gate\20260306-104712"
# csv_path = r"C:\Users\super\Downloads\own_datasets-20260306T115639Z-3-001\own_datasets\front_cam_gate\20260306-105523.csv"
# folder_path = r"C:\Users\super\Downloads\own_datasets-20260306T115639Z-3-001\own_datasets\Front_cam_try2\20260306-114544"
# csv_path = r"C:\Users\super\Downloads\own_datasets-20260306T115639Z-3-001\own_datasets\Front_cam_try2\20260306-114725.csv"

# ruben laptop
folder_path = r"/home/ruben/Downloads/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935"
csv_path = r"/home/ruben/Downloads/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142943.csv"
# folder_path = r"/home/ruben/Downloads/own_datasets/front_cam_gate/20260306-104712"
# csv_path = r"/home/ruben/Downloads/own_datasets/front_cam_gate/20260306-105523.csv"

# 2. Load and Prepare Data
df = pd.read_csv(csv_path)
image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

# Parameters for Flow
feature_params = dict(maxCorners=50, qualityLevel=0.000001, minDistance=20, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 3. Initialize
old_frame = ic.load_image(image_files[0])
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
last_t = get_img_time_from_filename(image_files[0])

w, h = old_gray.shape[::-1]

for image_file in image_files[1::]:
    frame = ic.load_image(image_file)
    if frame is None: break
    
    # --- SYNC LOGIC ---
    # Extract timestamp from filename (75044792 -> 75.044792)
    img_time = get_img_time_from_filename(image_file)
    dt = img_time - last_t
    last_t = img_time
    
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
    print(v)
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
            u = normalize_pixel_flow(np.array([dx, dy])) # measured flow in pixels
            u /= dt
            p = normalize_pixel_coords(np.array([a, b])) # pixel coordinate
            z = depth_from_flow(p, u, v, r)
            draw_point_with_depth(frame, (a, b), z, fac=10)

        p0 = good_new.reshape(-1, 1, 2)

    # Display Telemetry on screen
    # cv.putText(frame, f"Vel_X: {v_x:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv.putText(frame, f"Yaw_R: {yaw_rate:.2f} rad/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv.imshow('Sync Flow', frame)
    if cv.waitKey(int(dt*1000)) & 0xff == 27: break
    old_gray = frame_gray.copy()

cv.destroyAllWindows()