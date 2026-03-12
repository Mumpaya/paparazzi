import cv2
import numpy as np
import os
import glob
import re

DATASET_PATH = "/home/lapoveca/paparazzi/Bottom_cam_try/20260306-113610"

def get_image_list(folder_path):
    extensions = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files.sort(key=lambda f: [int(c) if c.isdigit() else c
                               for c in re.split(r'(\d+)', f)])
    return files

# --- Trackbar state ---
def nothing(x): pass

def main():
    images = get_image_list(DATASET_PATH)
    idx    = 0
    paused = True

    frame_ref = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and frame_ref[0] is not None:
            bgr_val = frame_ref[0][y, x]
            hsv_val = cv2.cvtColor(frame_ref[0][y:y+1, x:x+1], cv2.COLOR_BGR2HSV)[0][0]
            print(f"  Clicked px({x:3d},{y:3d})  BGR={bgr_val}  HSV={hsv_val}")

    # Main window with trackbars for live HSV tuning
    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H low",  "HSV Tuner",  18,  179, nothing)  # was 40
    cv2.createTrackbar("H high", "HSV Tuner",  76,  179, nothing)  # was 85
    cv2.createTrackbar("S low",  "HSV Tuner",  17,  255, nothing)  # was 15
    cv2.createTrackbar("S high", "HSV Tuner", 153,  255, nothing)  # was 255
    cv2.createTrackbar("V low",  "HSV Tuner", 124,  255, nothing)  # was 60
    cv2.createTrackbar("V high", "HSV Tuner", 255,  255, nothing)  # unchanged


    cv2.namedWindow("Original")
    cv2.setMouseCallback("Original", mouse_callback)

    print("Controls:")
    print("  SPACE     : pause / resume")
    print("  A / D     : step back / forward 1 frame")
    print("  Q / ESC   : quit")
    print("  LEFT CLICK on 'Original' window: print HSV of that pixel")
    print("  Tune trackbars in 'HSV Tuner' window to adjust mask live")
    print("  When happy, the final values are printed on quit.\n")

    while True:
        img_bgr = cv2.imread(images[idx])
        if img_bgr is None:
            idx += 1
            continue

        frame_ref[0] = img_bgr

        # Read trackbar values
        h_low  = cv2.getTrackbarPos("H low",  "HSV Tuner")
        h_high = cv2.getTrackbarPos("H high", "HSV Tuner")
        s_low  = cv2.getTrackbarPos("S low",  "HSV Tuner")
        s_high = cv2.getTrackbarPos("S high", "HSV Tuner")
        v_low  = cv2.getTrackbarPos("V low",  "HSV Tuner")
        v_high = cv2.getTrackbarPos("V high", "HSV Tuner")

        hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([h_low,  s_low,  v_low]),
                           np.array([h_high, s_high, v_high]))

        # Green tint overlay on masked pixels
        overlay = img_bgr.copy()
        overlay[mask > 0] = [0, 200, 0]
        display = cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)

        # HUD
        cv2.putText(display, f"Frame {idx}/{len(images)-1}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(display, f"H:[{h_low}-{h_high}] S:[{s_low}-{s_high}] V:[{v_low}-{v_high}]",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        if paused:
            cv2.putText(display, "PAUSED", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        cv2.imshow("Original", display)
        cv2.imshow("Mask",     mask)

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

    # Print final values on exit
    print("\n--- Final HSV range ---")
    print(f"hsv_lower = ({cv2.getTrackbarPos('H low','HSV Tuner')}, "
          f"{cv2.getTrackbarPos('S low','HSV Tuner')}, "
          f"{cv2.getTrackbarPos('V low','HSV Tuner')})")
    print(f"hsv_upper = ({cv2.getTrackbarPos('H high','HSV Tuner')}, "
          f"{cv2.getTrackbarPos('S high','HSV Tuner')}, "
          f"{cv2.getTrackbarPos('V high','HSV Tuner')})")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
