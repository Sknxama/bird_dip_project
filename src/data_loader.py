"""Load and merge features with CUB-200-2011 annotations."""

import os
import pandas as pd
from config import OUTPUTS_DIR, ANNOTATIONS_DIR


def load_annotations():
    """Load CUB annotation files and merge them."""
    # images.txt: image_id filepath
    images_df = pd.read_csv(
        os.path.join(ANNOTATIONS_DIR, "images.txt"),
        sep=" ",
        header=None,
        names=["image_id", "filepath"]
    )
    
    # image_class_labels.txt: image_id class_id
    labels_df = pd.read_csv(
        os.path.join(ANNOTATIONS_DIR, "image_class_labels.txt"),
        sep=" ",
        header=None,
        names=["image_id", "class_id"]
    )
    
    # classes.txt: class_id class_name
    classes_df = pd.read_csv(
        os.path.join(ANNOTATIONS_DIR, "classes.txt"),
        sep=" ",
        header=None,
        names=["class_id", "class_name"]
    )
    
    # Merge all
    merged = images_df.merge(labels_df, on="image_id")
    merged = merged.merge(classes_df, on="class_id")
    
    # Extract filename (e.g., "001.Black_footed_Albatross/xxx.jpg" -> "xxx.jpg")
    merged["filename"] = merged["filepath"].apply(lambda x: os.path.basename(x))
    
    return merged[["filename", "class_id", "class_name"]]


def load_features_with_labels(features_file="features.csv"):
    """
    Load extracted features and merge with class labels.
    
    Returns:
        X: DataFrame of numeric features
        y: Series of class_id labels
        meta: DataFrame with filename and class_name
    """
    features_path = os.path.join(OUTPUTS_DIR, features_file)
    features_df = pd.read_csv(features_path)
    
    labels_df = load_annotations()
    
    # Merge on filename
    merged = features_df.merge(labels_df, on="filename", how="left")
    merged = merged.dropna(subset=["class_id"])
    merged["class_id"] = merged["class_id"].astype(int)
    
    feature_cols = [
        "area", "perimeter", "circularity",
        "mean_h", "mean_s", "mean_v",
        "symmetry", "glcm_contrast", "glcm_energy", "fractal_dim"
    ]
    
    X = merged[feature_cols]
    y = merged["class_id"]
    meta = merged[["filename", "class_name", "folder"]]
    
    return X, y, meta