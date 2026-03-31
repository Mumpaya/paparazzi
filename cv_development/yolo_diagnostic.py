"""
YOLO Model Diagnostic
Test the trained model to understand its predictions
"""

import cv2
import os
from ultralytics import YOLO
import numpy as np

MODEL_PATH = '/home/roan2003/paparazzi/runs/detect/train2/weights/best.pt'
TEST_IMAGE = '/home/roan2003/paparazzi/cv_development/yolo_dataset/images/train/100011279.jpg'

print("=" * 70)
print("YOLO MODEL DIAGNOSTIC")
print("=" * 70)

# Load model
print(f"\nLoading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Check model info
print(f"\nModel Info:")
print(model.info())

# Load test image
image = cv2.imread(TEST_IMAGE)
print(f"\nTest image: {TEST_IMAGE}")
print(f"Image shape: {image.shape}")

# Try predictions with different confidence thresholds
thresholds = [0.01, 0.05, 0.1, 0.25, 0.5]

print(f"\n{'='*70}")
print("PREDICTIONS AT DIFFERENT CONFIDENCE THRESHOLDS")
print(f"{'='*70}")

for threshold in thresholds:
    print(f"\nThreshold: {threshold}")
    results = model.predict(TEST_IMAGE, conf=threshold, verbose=False)
    
    if results and len(results) > 0:
        detections = results[0]
        print(f"  Boxes detected: {len(detections.boxes) if hasattr(detections, 'boxes') else 0}")
        
        if hasattr(detections, 'boxes'):
            for i, box in enumerate(detections.boxes):
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                print(f"    Box {i+1}: Class={cls}, Confidence={conf:.4f}")

print(f"\n{'='*70}")
print("MODEL CLASSES")
print(f"{'='*70}")
print(f"Model classes: {model.names}")
