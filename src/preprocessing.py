"""Image preprocessing: grayscale, blur, histogram equalization."""

import cv2
import numpy as np
from config import GAUSSIAN_BLUR_KERNEL


def to_grayscale(bgr_image):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image):
    """Apply Gaussian blur to reduce fine noise."""
    return cv2.GaussianBlur(gray_image, GAUSSIAN_BLUR_KERNEL, 0)


def apply_histogram_equalization(gray_image):
    """Equalize histogram to improve contrast."""
    return cv2.equalizeHist(gray_image)


def preprocess(bgr_image):
    """
    Complete preprocessing pipeline.
    
    Input: BGR image (what cv2.imread returns)
    Output: tuple (gray, enhanced)
        - gray: original grayscale image
        - enhanced: improved image (blur + equalization)
    """
    gray = to_grayscale(bgr_image)
    blurred = apply_gaussian_blur(gray)
    enhanced = apply_histogram_equalization(blurred)
    return gray, enhanced