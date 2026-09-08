"""Test PCA visualization."""

from src.data_loader import load_features_with_labels
from src.pca_analysis import run_pca
from src.visualization import plot_pca_scatter

def main():
    print("Loading data...")
    X, y, meta = load_features_with_labels("features.csv")
    
    print("Running PCA...")
    pca_df = run_pca(X, meta, n_components=2)
    
    print("Creating scatter plot...")
    plot_pca_scatter(pca_df, output_name="pca_scatter.png", max_classes=25)
    
    print("\nDone! Open outputs/plots/pca_scatter.png to see the clusters.")

if __name__ == "__main__":
    main()