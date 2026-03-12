#!/usr/bin/env python3
"""
generate_test_images.py  –  Create 20+ synthetic test frames for the
obstacle-detector acceptance test.

Generates two categories:
  - mat_*   : mostly green carpet, score should be < 0.1
  - obs_*   : green carpet with obstacles (poles, walls, bright patches),
               score should be > 0.6

Images are 320×240 (representative bottom-camera resolution).
Place the output in test_images/ next to this script.
"""

import os
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV (cv2) is required.  Install with: pip install opencv-python")
    raise

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "test_images")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 320, 240
GREEN_BGR = (45, 140, 55)   # dark green carpet
GREEN_VAR = 12               # per-channel noise std

np.random.seed(42)


def make_green_base():
    """Return a noisy green-carpet image."""
    img = np.full((H, W, 3), GREEN_BGR, dtype=np.uint8)
    noise = np.random.normal(0, GREEN_VAR, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def save(img, name):
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, img)


# ── MAT-ONLY frames ──────────────────────────────────────────────────────────

for i in range(10):
    img = make_green_base()
    # Slight brightness variation per frame
    alpha = 0.9 + 0.2 * np.random.random()
    img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    save(img, f"mat_{i:02d}.png")

# Two more with slight texture (seams)
for i in range(2):
    img = make_green_base()
    y = np.random.randint(H // 4, 3 * H // 4)
    cv2.line(img, (0, y), (W, y), (40, 130, 50), 1)
    save(img, f"mat_seam_{i:02d}.png")


# ── OBSTACLE frames ──────────────────────────────────────────────────────────

# Vertical pole (orange cylinder)
for i in range(4):
    img = make_green_base()
    cx = np.random.randint(W // 4, 3 * W // 4)
    pole_w = np.random.randint(15, 35)
    color = (0, 100, 255)  # orange
    cv2.rectangle(img, (cx - pole_w // 2, 0), (cx + pole_w // 2, H), color, -1)
    # Add edge lines on the pole
    cv2.line(img, (cx - pole_w // 2, 0), (cx - pole_w // 2, H), (0, 0, 0), 2)
    cv2.line(img, (cx + pole_w // 2, 0), (cx + pole_w // 2, H), (0, 0, 0), 2)
    save(img, f"obs_pole_{i:02d}.png")

# Wall across the top half (ahead)
for i in range(3):
    img = make_green_base()
    wall_h = np.random.randint(H // 4, H // 2)
    color = (180, 180, 180)  # grey wall
    img[0:wall_h, :] = color
    # Add a horizontal line at the wall edge
    cv2.line(img, (0, wall_h), (W, wall_h), (60, 60, 60), 3)
    save(img, f"obs_wall_{i:02d}.png")

# Net / mesh pattern
for i in range(3):
    img = make_green_base()
    # Draw grid lines in front half
    for y in range(0, H // 2, 20):
        cv2.line(img, (0, y), (W, y), (30, 30, 30), 1)
    for x in range(0, W, 20):
        cv2.line(img, (x, 0), (x, H // 2), (30, 30, 30), 1)
    save(img, f"obs_net_{i:02d}.png")

# Bright obstacle blob
for i in range(2):
    img = make_green_base()
    cx = np.random.randint(W // 4, 3 * W // 4)
    cy = np.random.randint(H // 8, H // 3)
    r = np.random.randint(30, 60)
    cv2.circle(img, (cx, cy), r, (220, 220, 255), -1)
    cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
    save(img, f"obs_blob_{i:02d}.png")

print(f"Generated {len(os.listdir(OUT_DIR))} test images in {OUT_DIR}/")
