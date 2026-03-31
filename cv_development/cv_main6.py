import os
import cv2
import numpy as np
import pandas as pd
from obstacle_detector_v5 import ObstacleDetector

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "../AE4317_2019_datasets")

FLIGHT_FOLDER = os.path.join(DATASET_ROOT, "front_cam_gate/20260306-104712")
FLIGHT_CSV    = os.path.join(DATASET_ROOT, "front_cam_gate/20260306-105523.csv")

WINDOW_NAME = "ObstacleDetector V5"


def get_image_list(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return sorted([os.path.join(folder, f)
                   for f in os.listdir(folder) if f.lower().endswith(exts)])


def load_telemetry(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded telemetry: {df.shape[0]} rows from {os.path.basename(csv_path)}")
    return df


def match_telemetry_to_frame(df, img_path):
    try:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        img_t = int(fname) / 1e6
        idx   = (df['time'] - img_t).abs().argmin()
        return df.iloc[idx].to_dict()
    except (ValueError, KeyError):
        return None


def draw_hud(overlay, idx, total, fname, n_obs, n_cols, n_orange, n_hough, gap_str, paused, detector):
    """Burn dynamic info as text onto the overlay (top-left corner)."""
    lines = [
        f"Frame {idx+1}/{total}  {fname}",
        f"obs:{n_obs}/{n_cols}  orange:{n_orange}  hough:{n_hough}  {gap_str}",
        f"{'*** PAUSED ***' if paused else 'PLAYING'}  "
        f"F={'ON' if detector.use_flow else 'OFF'}  "
        f"H={'ON' if detector.use_hough else 'OFF'}  "
        f"O={'ON' if detector.use_orange else 'OFF'}  "
        f"G={'ON' if detector.use_ground else 'OFF'}",
        "SPC=pause  A/D=step  F/H/O/G=toggle  Q=quit",
    ]
    y = 18
    for line in lines:
        # Dark shadow for readability
        cv2.putText(overlay, line, (6, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(overlay, line, (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        y += 18


def main():
    images = get_image_list(FLIGHT_FOLDER)
    if not images:
        print(f"ERROR: No images found in {FLIGHT_FOLDER}")
        return

    flight_df = load_telemetry(FLIGHT_CSV)

    detector = ObstacleDetector(
        roi_top_frac    = 0.10,
        roi_bottom_frac = 0.90,
        clahe_clip_limit= 4.0,
        clahe_tile_grid = (8, 8),
        n_cols          = 22,
        flow_threshold  = 0.7,
        iir_alpha       = 0.35,
        max_corners     = 150,
        min_feature_dist= 4,
        refresh_interval= 8,
        flow_weight     = 1.5,
        hough_weight    = 1.0,
        orange_weight   = 2.5,
        hough_blur_kernel              = 5,
        hough_canny_low                = 50,
        hough_canny_high               = 150,
        hough_threshold                = 30,
        hough_min_length               = 40,
        hough_max_gap                  = 100,
        hough_vertical_tol_deg         = 20.0,
        hough_cluster_gap              = 35,
        hough_max_box_width            = 200,
        hough_min_aspect               = 1.5,
        hough_max_aspect               = 8.0,
        hough_border_margin            = 15,
        hough_max_interior_edge_density= 0.15,
        hsv_lower             = (0,  40,  50),
        hsv_upper             = (25, 255, 255),
        orange_min_area       = 300,
        orange_min_aspect     = 1.5,
        orange_min_confidence = 0.2,
        ground_weight         = 0.6,
        ground_green_lower    = (18, 17, 124),
        ground_green_upper    = (76, 153, 255),
        ground_morph_kernel   = (7, 7),
        ground_min_mean_green = 0.05,
    )

    print(f"\nLoaded {len(images)} frames.")
    print("Controls:  SPACE pause/resume  |  A/← back  |  D/→ forward  |  Q/ESC quit")
    print("           F=flow  H=hough  O=orange  G=ground")

    # Create the window once and keep it
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    idx    = 0
    paused = False

    while True:
        img_bgr = cv2.imread(images[idx])
        if img_bgr is None:
            idx = (idx + 1) % len(images)
            continue

        telem   = match_telemetry_to_frame(flight_df, images[idx])
        result  = detector.detect(img_bgr, telem=telem)
        overlay = detector.draw(img_bgr, result)

        # Scale to fit within 1920 px wide
        oh, ow = overlay.shape[:2]
        max_w  = 1920
        if ow > max_w:
            scale   = max_w / ow
            overlay = cv2.resize(overlay,
                                 (int(ow * scale), int(oh * scale)),
                                 interpolation=cv2.INTER_AREA)

        # Burn HUD info onto frame (no new window)
        fname   = os.path.basename(images[idx])
        n_obs   = len(result.obstacle_cols) if result.obstacle_cols else 0
        n_org   = len(result.orange_boxes)  if result.orange_boxes  else 0
        n_hgh   = len(result.hough_boxes)   if result.hough_boxes   else 0
        gap_str = f"gap={result.gap_center_x}" if result.gap_center_x is not None else "NO GAP"

        draw_hud(overlay, idx, len(images), fname,
                 n_obs, result.n_cols, n_org, n_hgh, gap_str, paused, detector)

        # Always show in the SAME window
        cv2.imshow(WINDOW_NAME, overlay)

        delay = 0 if paused else 30
        key   = cv2.waitKey(delay) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('d'), 83):       # D or →
            idx = (idx + 1) % len(images)
            detector._prev_roi = None
            detector._prev_pts = None
        elif key in (ord('a'), 81):       # A or ←
            idx = (idx - 1) % len(images)
            detector._prev_roi = None
            detector._prev_pts = None
        elif key == ord('f'):
            detector.use_flow   = not detector.use_flow
            print(f"Flow:   {'ON' if detector.use_flow   else 'OFF'}")
        elif key == ord('h'):
            detector.use_hough  = not detector.use_hough
            print(f"Hough:  {'ON' if detector.use_hough  else 'OFF'}")
        elif key == ord('o'):
            detector.use_orange = not detector.use_orange
            print(f"Orange: {'ON' if detector.use_orange else 'OFF'}")
        elif key == ord('g'):
            detector.use_ground = not detector.use_ground
            print(f"Ground: {'ON' if detector.use_ground else 'OFF'}")
        elif not paused:
            idx = (idx + 1) % len(images)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
