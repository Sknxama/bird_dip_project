"""Temporary test to verify feature extraction works."""

import cv2
import json
from pathlib import Path
from config import IMAGES_DIR
from src.preprocessing import preprocess
from src.segmentation import segment
from src.features import extract_all_features

# Find first image
image_files = list(Path(IMAGES_DIR).rglob("*.jpg"))
if not image_files:
    print("No images found")
    exit()

img = cv2.imread(str(image_files[0]))
name = image_files[0].name

# Pipeline
gray, enhanced = preprocess(img)
mask, bbox, contour = segment(enhanced)

if mask is None:
    print("Segmentation failed")
    exit()

features = extract_all_features(img, gray, mask, contour)

print(f"Image: {name}")
print("Extracted features:")
print(json.dumps(features, indent=2))

# Check: all values should be numbers, no None
all_valid = all(v is not None and isinstance(v, (int, float)) for v in features.values())
print(f"\nAll features are valid numbers: {all_valid}")