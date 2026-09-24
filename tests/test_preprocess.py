"""Temporary test to verify preprocessing works."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import os
from pathlib import Path
from config import IMAGES_DIR, PLOTS_DIR
from src.preprocessing import preprocess

# Find first image
image_files = list(Path(IMAGES_DIR).rglob("*.jpg"))
if not image_files:
    print("No images found")
    exit()

img = cv2.imread(str(image_files[0]))
gray, enhanced = preprocess(img)

# Save 3 versions for visual comparison
name = image_files[0].stem
cv2.imwrite(os.path.join(PLOTS_DIR, f"{name}_01_original.jpg"), img)
cv2.imwrite(os.path.join(PLOTS_DIR, f"{name}_02_gray.jpg"), gray)
cv2.imwrite(os.path.join(PLOTS_DIR, f"{name}_03_enhanced.jpg"), enhanced)

print(f"Saved 3 images to {PLOTS_DIR}:")
print("  1. original (color)")
print("  2. gray (grayscale)")
print("  3. enhanced (blur + equalization)")
print("Open them to compare.")