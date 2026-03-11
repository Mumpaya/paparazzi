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

if __name__ == "__main__":
    path = "Data/Playground" 
    images = get_full_dataset(path)
    
    cv2.namedWindow("1. Base Feed")
    cv2.namedWindow("2. Blue Mask")
    cv2.namedWindow("3. Contours / Blobs")
    cv2.namedWindow("4. Gate Detection")
    cv2.namedWindow("5. Camera + Gate")
    cv2.createTrackbar("Index", "4. Gate Detection", 0, max(0, len(images) - 1), lambda x: None)

    while True:
        idx = cv2.getTrackbarPos("Index", "4. Gate Detection")
        img_input = cv2.imread(images[idx])
        if img_input is None: continue

        # --- ROTATION ---
        frame = cv2.rotate(img_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = frame.shape[:2]

        # 1. COLOR MASKING
        # Step A — BGR pre-filter (blue channel dominant, caps on red & green)
        bgr_mask  = cv2.inRange(frame, CONF["BGR_LOW"], CONF["BGR_HIGH"])

        # Step B — HSV filter
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_mask  = cv2.inRange(hsv, CONF["BLUE_LOW"], CONF["BLUE_HIGH"])

        # Combined: pixel must pass BOTH filters
        blue_mask = cv2.bitwise_and(bgr_mask, hsv_mask)

        # Clean up noise — remove isolated specks
        open_k  = np.ones((CONF["OPEN_KERNEL"],  CONF["OPEN_KERNEL"]),  np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  open_k)

        # Bridge vertical gaps — tall narrow kernel merges split top/bottom pillar fragments
        close_k = np.ones((CONF["CLOSE_KERNEL"], 1), np.uint8)   # height × 1 px wide
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close_k)

        # 2. DETECTION
        pillar_x, accepted, rejected = find_blue_pillars(blue_mask, h, w)

        # --- Window 3: Contours / Blobs ---
        blob_view = frame.copy()

        # Accepted pillars → green outline + stats
        for cnt, (bx, by, bw, bh) in accepted:
            cv2.drawContours(blob_view, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(blob_view, (bx, by), (bx + bw, by + bh), (0, 220, 0), 1)
            ratio = bh / bw if bw > 0 else 0
            pct = 100.0 * bw * bh / (h * w)
            label = f"ar:{ratio:.2f} {pct:.1f}%"
            cv2.putText(blob_view, label, (bx, by - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(blob_view, label, (bx, by - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        # Rejected blobs → red outline + values
        for cnt, (bx, by, bw, bh) in rejected:
            cv2.drawContours(blob_view, [cnt], -1, (0, 0, 255), 1)
            cv2.rectangle(blob_view, (bx, by), (bx + bw, by + bh), (0, 0, 180), 1)
            ratio = bh / bw if bw > 0 else 0
            pct = 100.0 * bw * bh / (h * w)
            label = f":{ratio:.2f} {pct:.1f}%"
            cv2.putText(blob_view, label, (bx, by - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(blob_view, label, (bx, by - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Debug counter top-left
        cv2.putText(blob_view, f"ACC:{len(accepted)}  REJ:{len(rejected)}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blob_view, f"ACC:{len(accepted)}  REJ:{len(rejected)}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Window 4: Gate Detection ---
        gate_view = np.zeros((h, w, 3), dtype=np.uint8)
        gate_valid = False

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

        for x in pillar_x:
            cv2.line(gate_view, (x, 0), (x, h), (255, 0, 0), 3)

        if gate_valid:
            # Green centre line
            cv2.line(gate_view, (gate_cx, 0), (gate_cx, h), (0, 255, 0), 2)

            # Crosshair at gate centre
            cv2.drawMarker(gate_view, (gate_cx, gate_cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)

            # Info overlay
            cv2.putText(gate_view, "GATE DETECTED", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(gate_view, f"cx:{gate_cx}px  cy:{gate_cy}px", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(gate_view, f"cx:{100*gate_cx/w:.0f}%  cy:{100*gate_cy/h:.0f}%", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(gate_view, f"Angle: {angle_label}", (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_colour, 2)
            if angle_label == "GOOD" and gate_dist_m is not None:
                cv2.putText(gate_view, f"Dist: {gate_dist_m:.2f} m", (10, 116),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif len(pillar_x) == 2:
            cv2.putText(gate_view, "PILLARS: bad alignment", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # --- Window 5: Camera + Gate overlay ---
        # Draw the projected square gate onto the real camera feed.
        # Gate is assumed square → side length == horiz_dist (pillar centre separation).
        # The gate square spans: left edge = pillar_x[0], right edge = pillar_x[1],
        # and height == horiz_dist centred on gate_cy.
        cam_view = frame.copy()

        if gate_valid:
            gate_half = horiz_dist // 2
            sq_left   = pillar_x[0]
            sq_right  = pillar_x[1]
            sq_top    = max(0, gate_cy - gate_half)
            sq_bottom = min(h - 1, gate_cy + gate_half)

            # Gate square — cyan outline, thick
            cv2.rectangle(cam_view, (sq_left, sq_top), (sq_right, sq_bottom),
                          (255, 255, 0), 2)

            # Centre crosshair
            cv2.drawMarker(cam_view, (gate_cx, gate_cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 22, 2)

            # --- Side labels (small font, shadowed) ---
            def _txt(img, text, pt, colour, scale=0.38, thick=1):
                cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
                cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, colour, thick, cv2.LINE_AA)

            # Left side — distance (vertical, rotated text via column of chars)
            if angle_label == "GOOD" and gate_dist_m is not None:
                dist_str = f"{gate_dist_m:.2f}m"
                _txt(cam_view, dist_str, (max(0, sq_left - 38), gate_cy + 4),
                     (0, 255, 255))

            # Right side — heading angle label
            _txt(cam_view, angle_label,
                 (min(w - 45, sq_right + 4), gate_cy + 4), angle_colour)

            # Bottom — cx% / cy% position
            cx_pct = 100 * gate_cx / w
            cy_pct = 100 * gate_cy / h
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

        cv2.imshow("1. Base Feed",        frame)
        cv2.imshow("2. Blue Mask",        blue_mask)
        cv2.imshow("3. Contours / Blobs", blob_view)
        cv2.imshow("4. Gate Detection",   gate_view)
        cv2.imshow("5. Camera + Gate",    cam_view)

        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()