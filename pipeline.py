"""Batch pipeline: process all images and save features to CSV."""

import os
import csv
from pathlib import Path
from tqdm import tqdm
import cv2

from config import IMAGES_DIR, OUTPUTS_DIR, FEATURE_COLUMNS
from src.preprocessing import preprocess
from src.segmentation import segment
from src.features import extract_all_features


def process_single_image(image_path):
    """
    Process one image through the full pipeline.
    
    Returns: dict with features + metadata, or None if failed.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Step 1: Preprocess
    gray, enhanced = preprocess(img)
    
    # Step 2: Segment
    mask, bbox, contour = segment(enhanced)
    if mask is None:
        return None  # Segmentation failed
    
    # Step 3: Extract features
    features = extract_all_features(img, gray, mask, contour)
    
    # Add metadata
    features["filename"] = image_path.name
    features["folder"] = image_path.parent.name  # e.g. "001.Black_footed_Albatross"
    
    return features


def run_pipeline(limit=None, output_name="features.csv"):
    """
    Run batch processing on the dataset.
    
    Args:
        limit: if set, only process N images (for quick testing)
        output_name: name of the output CSV file
    """
    # Find all images
    image_files = list(Path(IMAGES_DIR).rglob("*.jpg"))
    total = len(image_files)
    print(f"Found {total} images in {IMAGES_DIR}")
    
    if limit:
        image_files = image_files[:limit]
        print(f"Processing first {limit} images for testing...")
    else:
        print(f"Processing all {total} images...")
    
    records = []
    failed = 0
    
    # Process with progress bar
    for img_path in tqdm(image_files, desc="Processing"):
        result = process_single_image(img_path)
        if result is not None:
            records.append(result)
        else:
            failed += 1
    
    # Save to CSV
    if not records:
        print("No images were successfully processed.")
        return
    
    output_path = os.path.join(OUTPUTS_DIR, output_name)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Determine fieldnames from first record
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"\nDone!")
    print(f"  Successfully processed: {len(records)}")
    print(f"  Failed: {failed}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    # Start with limit=50 for quick test, then remove limit for full run
    run_pipeline(limit=None)  # None = process all images