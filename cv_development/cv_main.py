import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import CameraCalibration
import cv2


# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT    = "../AE4317_2019_datasets"
CALIB_IMAGES    = os.path.join(DATASET_ROOT, "calibration_frontcam/20190121-163447")
CALIB_SAVE_PATH = "calibration_data.npz"
FLIGHT_FOLDER   = os.path.join(DATASET_ROOT, "sim_poles/20190121-160844")
FLIGHT_CSV    = os.path.join(DATASET_ROOT, "sim_poles/20190121-160857.csv")
POLES_CSV     = os.path.join(DATASET_ROOT, "sim_poles/pole_locations.csv")


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
samples = [image_files[0], image_files[len(image_files) // 2]]

# ── Display distorted vs undistorted ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Fisheye Distortion Correction", fontsize=14)

for col, fname in enumerate(samples):
    img_bgr  = cv2.imread(os.path.join(FLIGHT_FOLDER, fname))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    und_rgb  = cv2.cvtColor(cal.undistort(img_bgr), cv2.COLOR_BGR2RGB)

    axes[0][col].imshow(img_rgb)
    axes[0][col].set_title(f"Original: {fname}")
    axes[0][col].axis("off")

    axes[1][col].imshow(und_rgb)
    axes[1][col].set_title(f"Undistorted: {fname}")
    axes[1][col].axis("off")

plt.tight_layout()
plt.show()

