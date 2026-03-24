#!/usr/bin/env python3
"""Quick test of flyzone visualization"""
import subprocess
import time
import sys

# Run visualizer in background
print("Starting visualizer...")
proc = subprocess.Popen(
    ["./sw/airborne/modules/AF_8_V2_Improved/video_visualizer",
     "sw/airborne/modules/AF_8_V2_Improved/Playground"],
    cwd="/home/amampuya/AF/paparazzi",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Give it time to process frames
time.sleep(15)

# Kill it
proc.terminate()
try:
    proc.wait(timeout=2)
except subprocess.TimeoutExpired:
    proc.kill()

# Read output
stdout, stderr = proc.communicate()
print("Output:")
print(stdout)
if stderr:
    print("Errors:")
    print(stderr)

# Check for generated frames
import os
frames = sorted([f for f in os.listdir("/home/amampuya/AF/paparazzi") 
                 if f.startswith("viz_frame_real_") and f.endswith(".png")])
print(f"\n✅ Generated {len(frames)} frames")
for f in frames[:5]:
    print(f"  {f}")
