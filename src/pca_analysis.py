"""PCA analysis for dimensionality reduction."""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

from config import OUTPUTS_DIR, MODELS_DIR


def run_pca(X, meta, n_components=2):
    """
    Run PCA on feature matrix.
    
    Args:
        X: DataFrame of numeric features
        meta: DataFrame with filename, class_name, etc.
        n_components: number of PCA components (default 2 for plotting)
    
    Returns:
        pca_df: DataFrame with PC1, PC2, filename, class_name
    """
    # Standardize features (essential before PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Save models for later use
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_model.pkl"))
    
    # Create results DataFrame
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "filename": meta["filename"].values,
        "class_name": meta["class_name"].values,
    })
    
    # Save results
    output_path = os.path.join(OUTPUTS_DIR, "pca_results.csv")
    pca_df.to_csv(output_path, index=False)
    
    explained = pca.explained_variance_ratio_
    print(f"PCA completed.")
    print(f"  Explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}")
    print(f"  Total: {sum(explained):.3f}")
    print(f"  Results saved to: {output_path}")
    
    return pca_df