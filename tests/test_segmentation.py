"""Temporary test to verify segmentation works."""

import cv2
import os
import numpy as np
from pathlib import Path
from config import IMAGES_DIR, PLOTS_DIR
from src.preprocessing import preprocess
from src.segmentation import segment

# Find first image
image_files = list(Path(IMAGES_DIR).rglob("*.jpg"))
if not image_files:
    print("No images found")
    exit()

img = cv2.imread(str(image_files[0]))
name = image_files[0].stem

# Step 1: Preprocess
gray, enhanced = preprocess(img)

# Step 2: Segment
mask, bbox, contour = segment(enhanced)

if mask is None:
    print("Segmentation failed: no contour found")
    exit()

# Visualize results
x, y, w, h = bbox

# Original with bounding box drawn
img_bbox = img.copy()
cv2.rectangle(img_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Mask overlay on original
overlay = img.copy()
overlay[mask == 255] = (0, 0, 255)  # Red mask

# Save all
cv2.imwrite(os.path.join(PLOTS_DIR, f"{name}_04_enhanced.jpg"), enhanced)
cv2.imwrite(os.path.join(PLOTS_DIR, f"{name}_05_mask.jpg"), mask)
cv2.imwrite(os.path.join(PLOTS_DIR, f"{name}_06_bbox.jpg"), img_bbox)
cv2.imwrite(os.path.join(PLOTS_DIR, f"{name}_07_overlay.jpg"), overlay)

print(f"Saved 4 segmentation results to {PLOTS_DIR}:")
print("  4. enhanced (input to segmentation)")
print("  5. mask (white = bird, black = background)")
print("  6. bbox (green box around bird)")
print("  7. overlay (red mask on original)")
print("Check image 5: the bird should be white and background black.")