"""First-run script: generate all outputs from scratch."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("Bird Eco DIP - First Run Setup")
print("=" * 50)

print("\n[1/4] Processing CUB-200-2011 dataset...")
from pipeline import run_pipeline
run_pipeline(limit=None)

print("\n[2/4] Running PCA analysis...")
from src.data_loader import load_features_with_labels
from src.pca_analysis import run_pca
X, y, meta = load_features_with_labels("features.csv")
run_pca(X, meta, n_components=2)

print("\n[3/4] Training classifier...")
from src.classifier import train_classifier
model, accuracy = train_classifier(X, y, n_estimators=100)

print("\n[4/4] Building AVONET eco lookup...")
from src.avonet_cleaner import clean_avonet
clean_avonet()

from src.eco_mapper import build_eco_lookup_table
build_eco_lookup_table()

print("\n" + "=" * 50)
print("Setup complete! You can now run: streamlit run app.py")
print("=" * 50)