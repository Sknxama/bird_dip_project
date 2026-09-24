# Bird Eco DIP

**Bird Eco DIP** is a comprehensive digital image processing (DIP) system for analyzing bird plumage patterns and mapping them to ecological traits using classical computer vision techniques.

## Overview

This project processes the [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset (11,788 images, 200 species) using **OpenCV-based algorithms only** — no deep learning models (SAM, YOLO, etc.) are used for segmentation or feature extraction, as per project requirements.

The pipeline extracts 7 classical visual features (shape, color, texture, symmetry, fractal dimension), applies PCA for dimensionality reduction, trains a lightweight Random Forest classifier, and integrates ecological data from the [AVONET](https://figshare.com/s/b990722d72a26b5bfead) dataset.

## Features

- **Preprocessing**: Grayscale conversion, Gaussian blur, histogram equalization
- **Segmentation**: Otsu automatic thresholding + morphological operations (OpenCV only)
- **Feature Extraction**:
  - Shape: area, perimeter, circularity
  - Color: mean HSV values
  - Symmetry: bilateral correlation
  - Texture: GLCM contrast and energy
  - Complexity: fractal dimension (box-counting)
- **PCA Analysis**: 10D → 2D visualization with Plotly interactive scatter plot
- **Classification**: Random Forest lightweight classifier
- **Ecological Mapping**: AVONET traits (beak, wing, tarsus, tail length)
- **Web Interface**: Streamlit dashboard with 3 pages

## Project Structure

```
bird_dip_project/
├── app.py                    # Streamlit web interface
├── config.py                 # Global configuration and paths
├── pipeline.py               # Batch processing script
├── requirements.txt          # Python dependencies
├── src/
│   ├── preprocessing.py      # Image preprocessing (OpenCV)
│   ├── segmentation.py       # Otsu + morphology segmentation
│   ├── features.py           # Feature extraction (7 metrics)
│   ├── data_loader.py        # CUB-200-2011 annotation loader
│   ├── pca_analysis.py       # PCA dimensionality reduction
│   ├── classifier.py         # Random Forest training/prediction
│   ├── visualization.py      # Plotting utilities
│   ├── avonet_cleaner.py     # AVONET data cleaning
│   └── eco_mapper.py         # CUB ↔ AVONET mapping
├── data/
│   ├── images/               # CUB-200-2011 images (not in repo)
│   ├── annotations/          # CUB-200-2011 labels
│   └── avonet.csv            # AVONET ecological dataset
├── outputs/
│   ├── features.csv          # Extracted features (11,788 rows)
│   ├── pca_results.csv       # PCA projections
│   ├── eco_lookup.csv        # Ecological traits lookup table
│   └── models/               # Trained models (.pkl)
├── assets/
│   └── logo.png              # Project logo
└── tests/                    # Test scripts
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bird_dip_project.git
cd bird_dip_project
```

### 2. Create virtual environment

```
python -m venv venv
```

#### Windows

```
venv\Scripts\activate
```

#### Mac/Linux

```
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Download datasets

CUB-200-2011: Download from Caltech and extract to data/images/ and data/annotations/.
AVONET: Download from Figshare and save as data/avonet.csv.

### First-Time Setup (Reproducibility)

Since datasets and model weights are excluded from submission, run the following steps to generate all outputs:

### Step 1: Batch processing (generate features.csv)

```bash
python pipeline.py
```

This generates outputs/features.csv (11,788 rows of extracted features).

### Step 2: PCA and classifier training

```bash
python -c "from src.data_loader import load_features_with_labels; from src.pca_analysis import run_pca; from src.classifier import train_classifier; X, y, meta = load_features_with_labels('features.csv'); run_pca(X, meta); train_classifier(X, y)"
```

This generates:

- outputs/pca_results.csv (PCA projections)
- outputs/models/rf_model.pkl (trained Random Forest model)

### Step 3: AVONET ecological mapping

```bash
python -c "from src.avonet_cleaner import clean_avonet; from src.eco_mapper import build_eco_lookup_table; clean_avonet(); build_eco_lookup_table()"
```

This generates outputs/eco_lookup.csv (ecological traits lookup table).

### Step 4: Launch web interface

```bash
streamlit run app.py
```

Then open your browser at http://localhost:8501.
Note: All random operations use fixed seed SEED = 42 to ensure full reproducibility.

## Usage (Daily)

### Batch Processing

```bash
python pipeline.py
```

### Launch Web Interface

```bash
streamlit run app.py
```

Then open your browser at http://localhost:8501

## First-Time Setup (Reproducibility)

Since datasets and model weights are excluded from submission, run this once to generate all outputs:

```bash
python pipeline.py
```

Then train the classifier and build ecological mapping:

```bash
python -c "from src.data_loader import load_features_with_labels; from src.pca_analysis import run_pca; from src.classifier import train_classifier; X, y, meta = load_features_with_labels('features.csv'); run_pca(X, meta); train_classifier(X, y)"
```

Finally launch the app:

```bash
streamlit run app.py
```

All outputs will be generated in outputs/ directory.

## Key Parameters

| Parameter              | Value  | Description                     |
| ---------------------- | ------ | ------------------------------- |
| `SEED`                 | 42     | Random seed for reproducibility |
| `GAUSSIAN_BLUR_KERNEL` | (5, 5) | Noise reduction kernel          |
| `MORPH_KERNEL_SIZE`    | (5, 5) | Morphological cleaning kernel   |
| `PCA_COMPONENTS`       | 2      | Dimensions for visualization    |
| `RF_ESTIMATORS`        | 100    | Random Forest trees             |

## Technologies

- **Python 3.10+**
- **OpenCV (classical DIP operators)**
- **scikit-learn (PCA, Random Forest)**
- **scikit-image (GLCM texture)**
- **Streamlit (web interface)**
- **Plotly (interactive PCA visualization)**
- **Pandas / NumPy**

## Reproducibility

All random operations use SEED = 42. The full pipeline has been executed on 11,788 images and results are stored in outputs/.

## License

Academic project for Digital Image Processing course.
