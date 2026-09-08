"""Image segmentation using Otsu thresholding and morphology."""

import cv2
import numpy as np
from config import MORPH_KERNEL_SIZE


def apply_otsu_thresholding(gray_image):
    """
    Apply Otsu's automatic thresholding.
    Returns: binary mask (white = object, black = background)
    """
    # Otsu automatically finds the best threshold value
    _, binary = cv2.threshold(
        gray_image, 
        0, 
        255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def apply_morphology(binary_mask):
    """
    Clean the binary mask using morphological operations.
    Closing fills small holes, opening removes small noise.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    
    # Close: dilation followed by erosion (fills holes)
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Open: erosion followed by dilation (removes small noise)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opened


def find_largest_contour(cleaned_mask):
    """
    Find the largest contour in the mask.
    Assumption: the bird is the largest object in the image.
    Returns: mask_filled, bounding_box, contour
    """
    contours, _ = cv2.findContours(
        cleaned_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None, None, None
    
    # Keep the largest contour (the bird)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox = (x, y, w, h)
    
    # Create filled mask with only the largest contour
    mask_filled = np.zeros_like(cleaned_mask)
    cv2.drawContours(mask_filled, [largest_contour], -1, 255, -1)
    
    return mask_filled, bbox, largest_contour


def segment(enhanced_gray):
    """
    Complete segmentation pipeline.
    
    Input: enhanced grayscale image (from preprocessing)
    Output: tuple (mask, bbox, contour) or (None, None, None) if failed
        - mask: binary mask with only the bird (white)
        - bbox: (x, y, w, h) bounding box
        - contour: the contour points of the bird
    """
    binary = apply_otsu_thresholding(enhanced_gray)
    cleaned = apply_morphology(binary)
    result = find_largest_contour(cleaned)
    return result