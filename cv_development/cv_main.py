import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import CameraCalibration
import cv2
from pole_detector import PoleDetector
from orange_pole_detector import OrangePoleDetector
from green_ground_detector import GreenGroundDetector



# ── Paths ─────────────────────────────────────────────────────────────────────
# DATASET_ROOT    = "../AE4317_2019_datasets"
# CALIB_IMAGES    = os.path.join(DATASET_ROOT, "calibration_frontcam/20190121-163447")
# CALIB_SAVE_PATH = "calibration_data.npz"
# FLIGHT_FOLDER   = os.path.join(DATASET_ROOT, "sim_poles/20190121-160844")
# FLIGHT_CSV    = os.path.join(DATASET_ROOT, "sim_poles/20190121-160857.csv")
# POLES_CSV     = os.path.join(DATASET_ROOT, "sim_poles/pole_locations.csv")

# ── Paths (always relative to this script's location) ────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT    = os.path.join(SCRIPT_DIR, "../AE4317_2019_datasets")
CALIB_IMAGES    = os.path.join(DATASET_ROOT, "calibration_frontcam/20190121-163447")
CALIB_SAVE_PATH = os.path.join(SCRIPT_DIR, "calibration_data.npz")

# IN SIMULATION:
# FLIGHT_FOLDER   = os.path.join(DATASET_ROOT, "sim_poles/20190121-160844")
# FLIGHT_CSV      = os.path.join(DATASET_ROOT, "sim_poles/20190121-160857.csv")
# POLES_CSV       = os.path.join(DATASET_ROOT, "sim_poles/pole_locations.csv")

# IN CYBERZOO:
FLIGHT_FOLDER = os.path.join(DATASET_ROOT, "cyberzoo_poles/20190121-135009")
FLIGHT_CSV    = os.path.join(DATASET_ROOT, "cyberzoo_poles/20190121-135121.csv")  
POLES_CSV = os.path.join(DATASET_ROOT, "cyberzoo_poles/pole_locations.csv")

idx_pole = 205#225#142#187
idx_no_pole = 0

### DATASET EXPLORATION ####

# ── Load CSVs ─────────────────────────────────────────────────────────────────
flight_df = pd.read_csv(FLIGHT_CSV)
poles_df  = pd.read_csv(POLES_CSV)

print("=== Flight telemetry (20190121-160857.csv) ===")
print(f"Shape: {flight_df.shape}")
print(flight_df.head())

print("\n=== Pole locations (pole_locations.csv) ===")
print(f"Shape: {poles_df.shape}")
print(poles_df.head())

# ── Calibration: run once, then load from file ────────────────────────────────
cal = CameraCalibration(images_dir=CALIB_IMAGES)

if os.path.exists(CALIB_SAVE_PATH):
    print("Calibration file found, loading...")
    cal.load(CALIB_SAVE_PATH)
else:
    print("No calibration file found, running calibration...")
    cal.calibrate()
    cal.save(CALIB_SAVE_PATH)

# ── Pick two sample images ────────────────────────────────────────────────────
image_files = sorted(os.listdir(FLIGHT_FOLDER))
samples = [image_files[idx_no_pole], image_files[idx_pole]]
labels  = ["no pole", "pole"]

# ── Process images ────────────────────────────────────────────────────────────
processed = []
for fname in samples:
    img_bgr = cv2.imread(os.path.join(FLIGHT_FOLDER, fname))
    und_bgr = cal.undistort(img_bgr)
    rot_und_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rot_bgr = cv2.rotate(und_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    processed.append({
        "fname":    fname,
        "original": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "undist":   cv2.cvtColor(und_bgr,  cv2.COLOR_BGR2RGB),
        "rot_undist": cv2.cvtColor(rot_und_bgr, cv2.COLOR_BGR2RGB),
        "rotated":  cv2.cvtColor(rot_bgr,  cv2.COLOR_BGR2RGB),
    })

# ── Plot: 2 rows (samples) × 3 cols (stages) ─────────────────────────────────
# Images are tall & narrow after rotation → keep them small
fig, axes = plt.subplots(2, 4, figsize=(12, 7))
fig.suptitle("Original → Undistorted → Rotated Undistorted → Rotated", fontsize=13, y=1.01)

col_titles = ["Original", "Undistorted", "Rotated Undistorted", "Rotated"]
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=10)

for row, (data, label) in enumerate(zip(processed, labels)):
    axes[row][0].imshow(data["original"])
    axes[row][1].imshow(data["undist"])
    axes[row][2].imshow(data["rot_undist"])
    axes[row][3].imshow(data["rotated"])
    axes[row][0].set_ylabel(f"{label}\n{data['fname']}", fontsize=8, rotation=0,
                             labelpad=60, va="center")
    for col in range(4):
        axes[row][col].axis("off")

plt.tight_layout()
plt.show(block=False)


# detector = PoleDetector() original one, with edges and boxes
# detector    = PoleDetector(use_clahe=True,  canny_low=50, canny_high=150)
# detector_nc = PoleDetector(use_clahe=False, canny_low=50, canny_high=150)


# ── Pole detection on 4 frames ────────────────────────────────────────────────
detector     = PoleDetector(use_clahe=True, canny_low=50, canny_high=150)



detector = PoleDetector(
    use_clahe        = True,
    canny_low        = 50,
    canny_high       = 200,
    hough_min_length = 80,
    hough_max_gap    = 25,
    vertical_tol_deg = 40.0,
    cluster_gap      = 100,
    min_aspect_ratio = 2.5,
    max_aspect_ratio = 10.0,
    min_height_frac  = 0.30,
    border_margin    = 10,
    max_interior_edge_density = 0.10,
)

pole_indices = [142, 187, 205, 225]
pole_samples = [image_files[i] for i in pole_indices]

fig, axes = plt.subplots(4, 4, figsize=(12, 7))
fig.suptitle("Pole Detection — 4 sample frames", fontsize=11)

row_titles = ["Input", "CLAHE + Blurred", "Canny Edges", "Detections"]
for row, title in enumerate(row_titles):
    axes[row][0].set_ylabel(title, fontsize=9, rotation=90, labelpad=5, va="center")

for col, (fname, idx) in enumerate(zip(pole_samples, pole_indices)):
    img_bgr = cv2.imread(os.path.join(FLIGHT_FOLDER, fname))
    # und_bgr = cal.undistort(img_bgr)

    rot_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    result = detector.detect(rot_bgr)
    overlay = detector.draw(rot_bgr, result)
    
    axes[0][col].imshow(cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB))
    axes[1][col].imshow(result["blurred"], cmap="gray")
    axes[2][col].imshow(result["edges"],   cmap="gray")
    axes[3][col].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    axes[0][col].set_title(f"idx={idx}", fontsize=8)
    axes[3][col].set_xlabel(f"{len(result['boxes'])} pole(s)", fontsize=8)

    for row in range(4):
        axes[row][col].axis("off")

    print(f"[idx={idx}] {fname} → {len(result['boxes'])} box(es): {result['boxes']}")

plt.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.04, wspace=0.04, hspace=0.08)
plt.show(block=False)



detector     = OrangePoleDetector()
pole_indices = [142, 187, 205, 225]
pole_samples = [image_files[i] for i in pole_indices]

fig, axes = plt.subplots(4, 4, figsize=(12, 7))
fig.suptitle("Pole Detection — 4 sample frames", fontsize=11)

row_titles = ["Input", "CLAHE + Blurred", "Canny Edges", "Detections"]
for row, title in enumerate(row_titles):
    axes[row][0].set_ylabel(title, fontsize=9, rotation=90, labelpad=5, va="center")

for col, (fname, idx) in enumerate(zip(pole_samples, pole_indices)):
    img_bgr = cv2.imread(os.path.join(FLIGHT_FOLDER, fname))
    rot_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    result = detector.detect(rot_bgr)

    axes[0][col].imshow(cv2.cvtColor(rot_bgr,   cv2.COLOR_BGR2RGB))
    axes[1][col].imshow(result["blurred"], cmap="gray")
    axes[2][col].imshow(result["edges"],   cmap="gray")
    axes[3][col].imshow(cv2.cvtColor(detector.draw(rot_bgr, result),  cv2.COLOR_BGR2RGB))

    axes[0][col].set_title(f"idx={idx}", fontsize=8)
    axes[3][col].set_xlabel(f"{len(result['boxes'])} pole(s)", fontsize=8)

    for row in range(4):
        axes[row][col].axis("off")

    print(f"[idx={idx}] {fname} → {len(result['boxes'])} box(es): {result['boxes']}")

plt.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.04, wspace=0.04, hspace=0.08)
plt.show(block=False)




ground_det = GreenGroundDetector()
pole_indices = [142, 187, 205, 225]
pole_samples = [image_files[i] for i in pole_indices]

fig, axes = plt.subplots(3, 4, figsize=(14, 7))
fig.suptitle("Ground Detection — 4 sample frames", fontsize=11)

row_titles = ["Input", "Ground Mask", "Detection"]
for row, title in enumerate(row_titles):
    axes[row][0].set_ylabel(title, fontsize=9, rotation=90, labelpad=5, va="center")

for col, (fname, idx) in enumerate(zip(pole_samples, pole_indices)):
    img_bgr = cv2.imread(os.path.join(FLIGHT_FOLDER, fname))
    rot_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    result, debug_img = ground_det.detect(rot_bgr)

    axes[0][col].imshow(cv2.cvtColor(rot_bgr,   cv2.COLOR_BGR2RGB))
    axes[1][col].imshow(result.mask, cmap="gray")
    axes[2][col].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))

    axes[0][col].set_title(f"idx={idx}", fontsize=8)
    axes[2][col].set_xlabel(
        f"horizon={result.horizon_y}px  gnd={result.ground_frac:.1%}", fontsize=8
    )
    for row in range(3):
        axes[row][col].axis("off")

plt.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.04, wspace=0.04, hspace=0.08)
plt.show(block=False)



from ground_plane_detector_with_obstacles import ObstacleGroundPlaneDetector

ground_det   = ObstacleGroundPlaneDetector()
pole_indices = [142, 187, 205, 225]
pole_samples = [image_files[i] for i in pole_indices]

fig, axes = plt.subplots(4, 4, figsize=(14, 7))
fig.suptitle("Ground Detection — 4 sample frames", fontsize=11)

row_titles = ["Input", "Ground Mask", "Detection"]
for row, title in enumerate(row_titles):
    axes[row][0].set_ylabel(title, fontsize=9, rotation=90, labelpad=5, va="center")

for col, (fname, idx) in enumerate(zip(pole_samples, pole_indices)):
    img_bgr = cv2.imread(os.path.join(FLIGHT_FOLDER, fname))
    rot_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    result, debug_img = ground_det.detect(rot_bgr)

    axes[0][col].imshow(cv2.cvtColor(rot_bgr,       cv2.COLOR_BGR2RGB))
    axes[1][col].imshow(result.green_mask,  cmap="gray")   # raw green detections
    axes[2][col].imshow(result.floor_mask,  cmap="gray")   # interpolated full floor
    axes[3][col].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))


    axes[0][col].set_title(f"idx={idx}", fontsize=8)
    axes[2][col].set_xlabel(
        f"horizon={result.horizon_y}px  gnd={result.ground_frac:.1%}", fontsize=8
    )
    for row in range(3):
        axes[row][col].axis("off")

plt.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.04, wspace=0.04, hspace=0.08)
plt.show(block=False)



from ground_plane_estimator import GroundPlaneEstimator

estimator = GroundPlaneEstimator(K=cal.K, floor_z=1.0)

fig, axes = plt.subplots(3, 4, figsize=(14, 7))
fig.suptitle("Ground Plane Estimation — 4 sample frames", fontsize=11)

row_titles = ["Input", "Green Mask", "Plane Estimate"]
for row, title in enumerate(row_titles):
    axes[row][0].set_ylabel(title, fontsize=9, rotation=90, labelpad=5, va="center")

for col, (fname, idx) in enumerate(zip(pole_samples, pole_indices)):
    img_bgr = cv2.imread(os.path.join(FLIGHT_FOLDER, fname))
    rot_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    result, debug_img = estimator.detect(rot_bgr)

    axes[0][col].imshow(cv2.cvtColor(rot_bgr,   cv2.COLOR_BGR2RGB))
    axes[1][col].imshow(result.mask, cmap="gray")
    axes[2][col].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))

    axes[0][col].set_title(f"idx={idx}", fontsize=8)

    # Print plane stats under each detection image
    if result.plane_valid:
        n = result.normal
        xlabel = (f"n=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}]\n"
                  f"h={result.camera_height:.2f}m  gnd={result.ground_frac:.1%}")
    else:
        xlabel = "plane: FAILED"
    axes[2][col].set_xlabel(xlabel, fontsize=7)

    for row in range(3):
        axes[row][col].axis("off")

    print(f"[idx={idx}] valid={result.plane_valid}  "
          f"normal={result.normal}  height={result.camera_height}")

plt.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.06,
                    wspace=0.04, hspace=0.12)
plt.show(block=True)




# ── Run OrangePoleDetector on full flight as a video ─────────────────────────
detector = OrangePoleDetector()

image_files_sorted = sorted(os.listdir(FLIGHT_FOLDER))

# Simple video player using cv2.imshow
print("Starting video playback. Controls:")
print("  SPACE : pause / resume")
print("  LEFT  : step back 1 frame")
print("  RIGHT : step forward 1 frame")
print("  Q/ESC : quit")

idx     = 0
paused  = False
delay   = 30  # ms between frames (~33 fps)

while True:
    fname   = image_files_sorted[idx]
    img_bgr = cv2.imread(os.path.join(FLIGHT_FOLDER, fname))
    rot_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    result  = detector.detect(rot_bgr)
    overlay = detector.draw(rot_bgr, result)

    # HUD overlay
    n_poles = len(result["boxes"])
    h, w    = overlay.shape[:2]

    cv2.putText(overlay, f"Frame: {idx}/{len(image_files_sorted)-1}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(overlay, f"File:  {fname}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(overlay, f"Poles: {n_poles}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if n_poles == 0 else (0, 80, 255), 2)
    if paused:
        cv2.putText(overlay, "PAUSED", (w // 2 - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    cv2.imshow("Orange Pole Detector", overlay)

    key = cv2.waitKey(1 if not paused else 0) & 0xFF

    if key == ord('q') or key == 27:       # Q or ESC → quit
        break
    elif key == ord(' '):                   # SPACE → pause/resume
        paused = not paused
    elif key == 83 or key == ord('d'):      # RIGHT arrow or D → next frame
        idx = min(idx + 1, len(image_files_sorted) - 1)
    elif key == 81 or key == ord('a'):      # LEFT arrow or A → prev frame
        idx = max(idx - 1, 0)
    elif not paused:
        idx += 1
        if idx >= len(image_files_sorted):
            idx = 0  # loop back to start

cv2.destroyAllWindows()