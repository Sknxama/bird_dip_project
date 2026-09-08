"""PCA visualization and plotting utilities."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from config import PLOTS_DIR


def plot_pca_scatter(pca_df, output_name="pca_scatter.png", max_classes=30):
    """
    Create a scatter plot of PCA results.
    Colors points by bird family (derived from class_name).
    
    Args:
        pca_df: DataFrame with PC1, PC2, class_name
        output_name: filename for the saved plot
        max_classes: limit number of colors for readability
    """
    # Extract family from class_name (e.g., "001.Black_footed_Albatross" -> "Albatross")
    pca_df = pca_df.copy()
    pca_df["family"] = pca_df["class_name"].apply(
        lambda x: x.split(".")[-1].split("_")[-1] if "." in str(x) else "Unknown"
    )
    
    # Get top families by count
    top_families = pca_df["family"].value_counts().head(max_classes).index
    pca_df["plot_family"] = pca_df["family"].apply(
        lambda f: f if f in top_families else "Other"
    )
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    families = pca_df["plot_family"].unique()
    colors = plt.cm.tab20.colors
    
    for i, family in enumerate(families):
        subset = pca_df[pca_df["plot_family"] == family]
        plt.scatter(
            subset["PC1"], 
            subset["PC2"],
            c=[colors[i % len(colors)]],
            label=family,
            alpha=0.6,
            s=20,
        )
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Bird Image Features\n(Classical DIP + OpenCV)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"PCA scatter plot saved to: {output_path}")


def plot_feature_distribution(features_df, feature_name, output_name=None):
    """
    Plot histogram of a single feature.
    """
    if output_name is None:
        output_name = f"dist_{feature_name}.png"
    
    plt.figure(figsize=(8, 5))
    plt.hist(features_df[feature_name], bins=50, color="steelblue", edgecolor="black")
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title(f"Distribution of {feature_name}")
    plt.grid(axis="y", alpha=0.3)
    
    output_path = os.path.join(PLOTS_DIR, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Distribution plot saved to: {output_path}")