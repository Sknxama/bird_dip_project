"""Configuración global del proyecto."""

import os
import numpy as np

# ==================== SEMILLA ====================
SEED = 42
np.random.seed(SEED)

# ==================== RUTAS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

# Crear carpetas de salida si no existen
for folder in [OUTPUTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ==================== PARÁMETROS DIP ====================
GAUSSIAN_BLUR_KERNEL = (5, 5)
MORPH_KERNEL_SIZE = (5, 5)
OTSU_THRESHOLD = 0  # 0 = automático (Otsu decide)

# ==================== FEATURES ====================
FEATURE_COLUMNS = [
    "area",
    "perimeter",
    "circularity",
    "mean_h",
    "mean_s",
    "mean_v",
    "symmetry",
    "glcm_contrast",
    "glcm_energy",
    "fractal_dim",
]