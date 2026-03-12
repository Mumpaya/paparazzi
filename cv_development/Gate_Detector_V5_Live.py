import cv2
import numpy as np
import glob
import os
import re

# ==========================================================
# CONFIGURATION
# ==========================================================
CONF = {
    # HSV Blue Range
    "BLUE_LOW":  np.array([79, 90, 120]),
    "BLUE_HIGH": np.array([137, 220, 255]),
    # BGR pre-filter — (B, G, R) order
    "BGR_LOW":  np.array([70,   17,   20]),
    "BGR_HIGH": np.array([255, 200, 150]),
    "MIN_BBOX_FRAC": 0.002,      
    "MIN_ASPECT_RATIO": 2.27,
    "OPEN_KERNEL":  3,
    "CLOSE_KERNEL": 11,
    # Gate validation thresholds
    "GATE_MAX_VERT_DIFF":  0.20,  # max vertical centre diff as fraction of frame height
    "GATE_MIN_HORIZ_DIST": 0.10,  # min horizontal centre dist as fraction of frame width
    "GATE_ANGLE_THRESH":   0.5,  # if one pillar bbox area is >20% larger → LEFT or RIGHT label
    # Distance estimation — calibrate: if horiz_dist == frame width → this distance in metres
    "DIST_CALIB_M":        0.79,   # metres when gate spans full frame width
}

def get_full_dataset(folder_path):
    files = [f for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp") for f in glob.glob(os.path.join(folder_path, ext))]
    files.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', f)])
    return files

def find_blue_pillars(mask, img_h, img_w):
    """Finds large vertical blue contours.
    Returns:
        pillar_x    – list of X centre positions of accepted pillars
        accepted    – list of (contour, bounding_rect) that passed all filters
        rejected    – list of (contour, bounding_rect) that failed area/aspect filters
    """
    min_bbox_px = CONF["MIN_BBOX_FRAC"] * img_h * img_w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pillars_x = []
    accepted = []
    rejected = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = w * h
        aspect_ratio = h / w if w > 0 else 0
        area_ok  = bbox_area > min_bbox_px
        aspect_ok = aspect_ratio >= CONF["MIN_ASPECT_RATIO"]
        if area_ok and aspect_ok:
            pillars_x.append(x + w // 2)
            accepted.append((cnt, (x, y, w, h)))
        else:
            rejected.append((cnt, (x, y, w, h)))

    # Pick the pair of accepted blobs with the greatest horizontal separation.
    # This is more robust than "2 largest" — real gate pillars are always far apart in X,
    # whereas spurious blobs tend to cluster near a real pillar.
    best_pair = []
    best_dist = -1
    for i in range(len(accepted)):
        for j in range(i + 1, len(accepted)):
            xi = accepted[i][1][0] + accepted[i][1][2] // 2
            xj = accepted[j][1][0] + accepted[j][1][2] // 2
            dist = abs(xi - xj)
            if dist > best_dist:
                best_dist = dist
                best_pair = sorted([accepted[i], accepted[j]], key=lambda c: c[1][0])

    results = [item[1][0] + item[1][2] // 2 for item in best_pair]
    return results, accepted, rejected

# ==========================================================
# CORE PROCESSING — callable from a control node
# ==========================================================
def process_frame(frame):
    """Run full gate detection pipeline on a single BGR frame (already rotated).

    Returns
    -------
    gate_state : dict
        Always present keys:
            "detected"   : bool   — True if a valid gate was found
        When detected is True:
            "gate_cx_px" : int    — gate centre X in pixels
            "gate_cy_px" : int    — gate centre Y in pixels
            "gate_cx_pct": float  — gate centre X as % of frame width  (0–100)
            "gate_cy_pct": float  — gate centre Y as % of frame height (0–100)
            "angle"      : str    — "GOOD" | "LEFT" | "RIGHT"
            "dist_m"     : float | None — estimated distance in metres (None if angle != GOOD)
            "sq_left"    : int    — left  edge of projected gate square (px)
            "sq_right"   : int    — right edge of projected gate square (px)
            "sq_top"     : int    — top   edge of projected gate square (px)
            "sq_bottom"  : int    — bottom edge of projected gate square (px)
        When detected is False:
            all above keys are absent.
    cam_view : np.ndarray
        Annotated BGR image (same size as frame) for display.
    """
    h, w = frame.shape[:2]

    # 1. COLOR MASKING
    # Step A — BGR pre-filter (blue channel dominant, caps on red & green)
    bgr_mask = cv2.inRange(frame, CONF["BGR_LOW"], CONF["BGR_HIGH"])

    # Step B — HSV filter
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, CONF["BLUE_LOW"], CONF["BLUE_HIGH"])

    # Combined: pixel must pass BOTH filters
    blue_mask = cv2.bitwise_and(bgr_mask, hsv_mask)

    # Clean up noise — remove isolated specks
    open_k    = np.ones((CONF["OPEN_KERNEL"], CONF["OPEN_KERNEL"]), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  open_k)

    # Bridge vertical gaps — tall narrow kernel merges split top/bottom pillar fragments
    close_k   = np.ones((CONF["CLOSE_KERNEL"], 1), np.uint8)   # height × 1 px wide
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close_k)

    # 2. DETECTION
    pillar_x, accepted, rejected = find_blue_pillars(blue_mask, h, w)

    # 3. GATE VALIDATION
    gate_valid  = False
    gate_state  = {"detected": False}
    angle_label = None
    angle_colour = (0, 165, 255)
    gate_dist_m  = None
    gate_cx = gate_cy = 0
    horiz_dist = 0
    sq_left = sq_right = sq_top = sq_bottom = 0

    if len(pillar_x) == 2:
        # Find the accepted bbox whose centre X matches each pillar_x
        def find_bbox(cx_target):
            for cnt, (bx, by, bw, bh) in accepted:
                if bx + bw // 2 == cx_target:
                    return (bx, by, bw, bh)
            return None

        bbox0 = find_bbox(pillar_x[0])
        bbox1 = find_bbox(pillar_x[1])

        if bbox0 and bbox1:
            bx0, by0, bw0, bh0 = bbox0
            bx1, by1, bw1, bh1 = bbox1

            cy0 = by0 + bh0 // 2   # vertical centre of left pillar bbox
            cy1 = by1 + bh1 // 2   # vertical centre of right pillar bbox

            vert_diff  = abs(cy0 - cy1)
            horiz_dist = abs(pillar_x[1] - pillar_x[0])

            vert_ok  = vert_diff  < CONF["GATE_MAX_VERT_DIFF"]  * h
            horiz_ok = horiz_dist > CONF["GATE_MIN_HORIZ_DIST"] * w

            gate_valid = vert_ok and horiz_ok

            if gate_valid:
                # --- Gate centre position ---
                gate_cx = (pillar_x[0] + pillar_x[1]) // 2        # horizontal centre
                gate_cy = (cy0 + cy1) // 2                         # vertical centre

                # --- Approach angle (LEFT / RIGHT / GOOD) ---
                area0 = bw0 * bh0   # left pillar bbox area
                area1 = bw1 * bh1   # right pillar bbox area
                thresh = CONF["GATE_ANGLE_THRESH"]
                if area0 > area1 * (1 + thresh):
                    angle_label = "LEFT"
                    angle_colour = (0, 165, 255)   # orange
                elif area1 > area0 * (1 + thresh):
                    angle_label = "RIGHT"
                    angle_colour = (0, 165, 255)
                else:
                    angle_label = "GOOD"
                    angle_colour = (0, 255, 0)

                # --- Distance estimate (only meaningful when angle is GOOD) ---
                # Linear model: distance ∝ 1/horiz_dist
                # At horiz_dist == w  →  DIST_CALIB_M metres
                gate_dist_m = CONF["DIST_CALIB_M"] * w / horiz_dist if horiz_dist > 0 else None

                # --- Projected gate square (gate is square → side == horiz_dist) ---
                gate_half = horiz_dist // 2
                sq_left   = pillar_x[0]
                sq_right  = pillar_x[1]
                sq_top    = max(0, gate_cy - gate_half)
                sq_bottom = min(h - 1, gate_cy + gate_half)

                gate_state = {
                    "detected"   : True,
                    "gate_cx_px" : gate_cx,
                    "gate_cy_px" : gate_cy,
                    "gate_cx_pct": round(100.0 * gate_cx / w, 1),
                    "gate_cy_pct": round(100.0 * gate_cy / h, 1),
                    "angle"      : angle_label,
                    "dist_m"     : gate_dist_m,
                    "sq_left"    : sq_left,
                    "sq_right"   : sq_right,
                    "sq_top"     : sq_top,
                    "sq_bottom"  : sq_bottom,
                }

    # 4. CAMERA + GATE OVERLAY (Window 5 logic — unchanged)
    cam_view = frame.copy()

    # --- Side label helper (shadowed text) ---
    def _txt(img, text, pt, colour, scale=0.38, thick=1):
        cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thick, cv2.LINE_AA)

    if gate_valid:
        # Gate square — cyan outline, thick
        cv2.rectangle(cam_view, (sq_left, sq_top), (sq_right, sq_bottom),
                      (255, 255, 0), 2)

        # Centre crosshair
        cv2.drawMarker(cam_view, (gate_cx, gate_cy), (0, 255, 255),
                       cv2.MARKER_CROSS, 22, 2)

        # Left side — distance
        if angle_label == "GOOD" and gate_dist_m is not None:
            dist_str = f"{gate_dist_m:.2f}m"
            _txt(cam_view, dist_str, (max(0, sq_left - 38), gate_cy + 4),
                 (0, 255, 255))

        # Right side — heading angle label
        _txt(cam_view, angle_label,
             (min(w - 45, sq_right + 4), gate_cy + 4), angle_colour)

        # Bottom — cx% / cy% position
        cx_pct  = 100 * gate_cx / w
        cy_pct  = 100 * gate_cy / h
        pos_str = f"x:{cx_pct:.0f}% y:{cy_pct:.0f}%"
        _txt(cam_view, pos_str,
             (max(0, gate_cx - 32), min(h - 4, sq_bottom + 12)),
             (200, 200, 200))

    else:
        # No valid gate detected
        cv2.putText(cam_view, "No Detection", (w // 2 - 55, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(cam_view, "No Detection", (w // 2 - 55, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return gate_state, cam_view


# ==========================================================
# LIVE SIMULATION — iterates dataset as if it were a live feed
# ==========================================================
if __name__ == "__main__":
    # Simulated frame rate: ~30 fps playback (33 ms per frame).
    # Press 'q' to quit, SPACE to pause/resume.
    PLAYBACK_FPS = 30
    FRAME_DELAY_MS = max(1, 1000 // PLAYBACK_FPS)

    path   = "Data/Playground"
    images = get_full_dataset(path)

    cv2.namedWindow("Gate View")
    paused = False
    idx    = 0

    while True:
        if not paused:
            img_input = cv2.imread(images[idx % len(images)])
            if img_input is not None:
                # --- ROTATION (same as before) ---
                frame = cv2.rotate(img_input, cv2.ROTATE_90_COUNTERCLOCKWISE)

                gate_state, cam_view = process_frame(frame)

                # gate_state is ready to be consumed by a control node, e.g.:
                #   if gate_state["detected"]:
                #       send_to_controller(gate_state)
                print(gate_state)   # visible in terminal; remove/replace in production

                cv2.imshow("Gate View", cam_view)
            idx += 1

        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused

    cv2.destroyAllWindows()