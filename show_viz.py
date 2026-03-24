#!/usr/bin/env python3
"""Display visualization frames in a loop"""
import os
import time
import subprocess
import sys

# Get all visualization frames
frames = sorted([f for f in os.listdir('/home/amampuya/AF/paparazzi') 
                 if f.startswith('test_main_') and f.endswith('.png')])

if not frames:
    print("No frames found!")
    sys.exit(1)

print(f"Found {len(frames)} frames. Creating slideshow...")
print("Looping all frames - Press Ctrl+C to stop\n")

# Create a simple HTML viewer or use feh/eog
frame_path = '/home/amampuya/AF/paparazzi/'

try:
    # Try to use 'eog' (Eye of GNOME) to view images
    for i, frame in enumerate(frames * 3):  # Loop 3 times
        filepath = os.path.join(frame_path, frame)
        print(f"\r[{i % len(frames) + 1}/{len(frames)}] {frame}", end='', flush=True)
        
        # Use 'feh' or 'display' to show image
        proc = subprocess.Popen(['feh', '-x', '-d', '--fullscreen', filepath],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        
        time.sleep(2)  # Show each frame for 2 seconds
        proc.terminate()
        
except KeyboardInterrupt:
    print("\n\nStopped!")
    sys.exit(0)
except FileNotFoundError:
    print("\nfeh not found. Trying alternative viewer...")
    # Fallback to convert/display
    try:
        for i, frame in enumerate(frames * 3):
            filepath = os.path.join(frame_path, frame)
            print(f"\r[{i % len(frames) + 1}/{len(frames)}] {frame}", end='', flush=True)
            
            proc = subprocess.Popen(['display', filepath],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            time.sleep(2)
            proc.terminate()
    except KeyboardInterrupt:
        print("\n\nStopped!")
    except:
        print("\nNo image viewer found. Listing frames instead:\n")
        for i, frame in enumerate(frames):
            print(f"  {i+1:2d}. {frame}")
