import cv2
import os
import glob
import re

from ground_edge_detector import GroundEdgeDetector

DATASET_PATH = "/home/lapoveca/paparazzi/Bottom_cam_try/20260306-113610"
def get_image_list(folder_path: str) -> list[str]:
    """Return all images in folder, sorted by numeric filename."""
    extensions = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files.sort(key=lambda f: [int(c) if c.isdigit() else c
                               for c in re.split(r'(\d+)', f)])
    return files


def main():
    images = get_image_list(DATASET_PATH)
    detector = GroundEdgeDetector()

    print("Starting video playback. Controls:")
    print("  SPACE : pause / resume")
    print("  LEFT  : step back 1 frame")
    print("  RIGHT : step forward 1 frame")
    print("  Q/ESC : quit")

    idx    = 0
    paused = False
    delay  = 30

    while True:
        fname   = images[idx]
        img_bgr = cv2.imread(images[idx])
        if img_bgr is None:
            idx += 1
            continue

        result  = detector.detect(img_bgr)
        overlay = detector.draw(img_bgr, result)

        n_poles = len(result.contours)

        h, w    = overlay.shape[:2]

        n_confirmed = len(result.confirmed_lines)
        
        cv2.putText(overlay, f"Confirmed edges: {n_confirmed}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255) if n_confirmed > 0 else (100, 100, 100), 2)

        cv2.putText(overlay, f"Frame: {idx}/{len(images)-1}",        # ← fixed
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"File:  {fname}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, f"Regions: {n_poles}",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0) if n_poles == 0 else (0, 80, 255), 2)

        if paused:
            cv2.putText(overlay, "PAUSED", (w // 2 - 50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        cv2.imshow("Green Edge Detector", overlay)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 83 or key == ord('d'):
            idx = min(idx + 1, len(images) - 1)         # ← fixed
        elif key == 81 or key == ord('a'):
            idx = max(idx - 1, 0)
        elif not paused:
            idx += 1
            if idx >= len(images):                       # ← fixed
                idx = 0

    cv2.destroyAllWindows()   # ← moved OUTSIDE the while loop





if __name__ == "__main__":
    main()