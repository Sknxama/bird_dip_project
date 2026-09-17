"""Temporary test to verify PCA works."""

from src.data_loader import load_features_with_labels
from src.pca_analysis import run_pca

def main():
    print("Loading features and labels...")
    X, y, meta = load_features_with_labels("features.csv")
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    print("\nRunning PCA...")
    pca_df = run_pca(X, meta, n_components=2)
    
    print(f"\nPCA results preview:")
    print(pca_df.head())

if __name__ == "__main__":
    main()