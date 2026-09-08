"""Feature extraction from segmented bird images."""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_shape_features(contour):
    """
    Extract shape features from contour.
    Returns: dict with area, perimeter, circularity
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Circularity: 1.0 = perfect circle, <1.0 = less circular
    if perimeter > 0:
        circularity = 4.0 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0.0
    
    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
    }


def extract_color_features(bgr_image, mask):
    """
    Extract average color in HSV space within the mask region.
    Returns: dict with mean_h, mean_s, mean_v
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    # cv2.mean returns (H, S, V, alpha) when mask is provided
    mean_hsv = cv2.mean(hsv, mask=mask)
    
    return {
        "mean_h": float(mean_hsv[0]),
        "mean_s": float(mean_hsv[1]),
        "mean_v": float(mean_hsv[2]),
    }


def extract_symmetry(gray_image, mask):
    """
    Measure bilateral symmetry by correlating left and right halves.
    Returns: dict with symmetry score (-1 to 1, higher = more symmetric)
    """
    # Focus only on the masked region for cleaner comparison
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    
    h, w = masked_gray.shape
    mid = w // 2
    
    left = masked_gray[:, :mid]
    right = masked_gray[:, -mid:]
    
    # Flip right half horizontally to compare with left
    right_flipped = cv2.flip(right, 1)
    
    # Ensure same dimensions
    min_w = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_w]
    right_flipped = right_flipped[:, :min_w]
    
    # Template matching gives correlation score
    result = cv2.matchTemplate(left, right_flipped, cv2.TM_CCOEFF_NORMED)
    symmetry_score = float(result[0][0])
    
    return {
        "symmetry": symmetry_score,
    }


def extract_texture_glcm(gray_image):
    """
    Extract texture features using Gray Level Co-occurrence Matrix (GLCM).
    Returns: dict with glcm_contrast, glcm_energy
    """
    # Reduce to 32 gray levels for faster computation
    gray32 = (gray_image // 8).astype(np.uint8)
    
    glcm = graycomatrix(
        gray32,
        distances=[1],
        angles=[0],
        levels=32,
        symmetric=True,
        normed=True,
    )
    
    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    
    return {
        "glcm_contrast": float(contrast),
        "glcm_energy": float(energy),
    }


def extract_fractal_dimension(gray_image, mask):
    """
    Estimate fractal dimension using a simplified box-counting method
    on the edges within the masked region.
    Returns: dict with fractal_dim
    """
    # Detect edges within the bird region only
    edges = cv2.Canny(gray_image, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Count edge pixels
    edge_pixels = np.sum(edges > 0)
    
    if edge_pixels == 0:
        return {"fractal_dim": 0.0}
    
    # Get bounding box of edge region
    ys, xs = np.where(edges > 0)
    if len(xs) < 2:
        return {"fractal_dim": 0.0}
    
    Ly = ys.max() - ys.min() + 1
    Lx = xs.max() - xs.min() + 1
    L = max(Lx, Ly)
    
    if L <= 1:
        return {"fractal_dim": 0.0}
    
    # Simplified box-counting approximation
    fractal_dim = np.log(edge_pixels) / np.log(L)
    
    return {
        "fractal_dim": float(fractal_dim),
    }


def extract_all_features(bgr_image, gray_image, mask, contour):
    """
    Complete feature extraction pipeline.
    
    Input:
        bgr_image: original color image
        gray_image: original grayscale image
        mask: binary mask of the bird
        contour: contour of the bird
    Output:
        dict with all 7+ features
    """
    features = {}
    
    # 1-3. Shape features
    features.update(extract_shape_features(contour))
    
    # 4. Color features
    features.update(extract_color_features(bgr_image, mask))
    
    # 5. Symmetry
    features.update(extract_symmetry(gray_image, mask))
    
    # 6. Texture (GLCM)
    features.update(extract_texture_glcm(gray_image))
    
    # 7. Fractal dimension
    features.update(extract_fractal_dimension(gray_image, mask))
    
    return features