import cv2
import numpy as np
import os
import glob
import re
from collections import deque

DATASET_PATH = "/home/lapoveca/paparazzi/Bottom_cam_try/20260306-113610"

def get_image_list(folder_path):
    extensions = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files.sort(key=lambda f: [int(c) if c.isdigit() else c
                               for c in re.split(r'(\d+)', f)])
    return files

def nothing(x): pass

def main():
    images = get_image_list(DATASET_PATH)
    idx    = 0
    paused = True

    # Fixed HSV from your calibration
    hsv_lower = np.array([18,  17, 124])
    hsv_upper = np.array([76, 153, 255])

    cv2.namedWindow("Hough Tuner")
    cv2.createTrackbar("Threshold",   "Hough Tuner", 100, 300, nothing)
    cv2.createTrackbar("Min Length",  "Hough Tuner",  80, 300, nothing)
    cv2.createTrackbar("Max Gap",     "Hough Tuner",  20, 100, nothing)
    cv2.createTrackbar("Blur",        "Hough Tuner",   7,  21, nothing)
    cv2.createTrackbar("Morph Close", "Hough Tuner",   5,  31, nothing)

    print("Controls:")
    print("  SPACE   : pause / resume")
    print("  A / D   : step frames")
    print("  Q / ESC : quit and print final values")

    while True:
        img_bgr = cv2.imread(images[idx])
        if img_bgr is None:
            idx += 1
            continue

        # Read trackbars
        thresh     = cv2.getTrackbarPos("Threshold",   "Hough Tuner")
        min_len    = cv2.getTrackbarPos("Min Length",  "Hough Tuner")
        max_gap    = cv2.getTrackbarPos("Max Gap",     "Hough Tuner")
        blur_k     = cv2.getTrackbarPos("Blur",        "Hough Tuner")
        morph_k    = cv2.getTrackbarPos("Morph Close", "Hough Tuner")

        # Ensure odd kernel sizes
        blur_k  = max(1, blur_k  | 1)
        morph_k = max(1, morph_k | 1)

        # --- Pipeline ---
        blurred  = cv2.GaussianBlur(img_bgr, (blur_k, blur_k), 0)
        hsv      = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask     = cv2.inRange(hsv, hsv_lower, hsv_upper)

        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        eroded   = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary = cv2.subtract(mask, eroded)

        raw_lines = cv2.HoughLinesP(
            boundary, rho=1, theta=np.pi / 180,
            threshold=max(1, thresh),
            minLineLength=max(1, min_len),
            maxLineGap=max_gap,
        )

        # Draw
        overlay = img_bgr.copy()
        overlay[mask > 0] = [0, 200, 0]
        out = cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)

        n_lines = 0
        if raw_lines is not None:
            n_lines = len(raw_lines)
            for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
                cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # HUD
        cv2.putText(out, f"Frame {idx}/{len(images)-1}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(out, f"Lines detected: {n_lines}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(out, f"thr={thresh} minL={min_len} gap={max_gap}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1)
        if paused:
            cv2.putText(out, "PAUSED", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        cv2.imshow("Hough Tuner", out)
        cv2.imshow("Boundary",   boundary)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d') or key == 83:
            idx = min(idx + 1, len(images) - 1)
            paused = True
        elif key == ord('a') or key == 81:
            idx = max(idx - 1, 0)
            paused = True
        elif not paused:
            idx += 1
            if idx >= len(images):
                idx = 0

    print("\n--- Final Hough parameters ---")
    print(f"hough_threshold  = {cv2.getTrackbarPos('Threshold',  'Hough Tuner')}")
    print(f"hough_min_length = {cv2.getTrackbarPos('Min Length', 'Hough Tuner')}")
    print(f"hough_max_gap    = {cv2.getTrackbarPos('Max Gap',    'Hough Tuner')}")
    print(f"blur_ksize       = {cv2.getTrackbarPos('Blur',       'Hough Tuner') | 1}")
    print(f"morph_ksize      = {cv2.getTrackbarPos('Morph Close','Hough Tuner') | 1}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
