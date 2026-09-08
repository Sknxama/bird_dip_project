"""Diagnose data quality issues."""

import pandas as pd
import numpy as np
from src.data_loader import load_features_with_labels

def main():
    X, y, meta = load_features_with_labels("features.csv")
    
    print("=" * 50)
    print("DIAGNOSIS REPORT")
    print("=" * 50)
    
    print(f"\n1. Samples loaded: {len(X)}")
    print(f"2. Unique species: {y.nunique()}")
    print(f"3. Features shape: {X.shape}")
    
    print(f"\n4. NaN values in features:")
    print(X.isnull().sum())
    
    print(f"\n5. Infinite values in features:")
    print(np.isinf(X).sum())
    
    print(f"\n6. Feature statistics (min/max/mean):")
    print(X.describe().loc[['min', 'max', 'mean']])
    
    print(f"\n7. Class distribution (top 10):")
    print(y.value_counts().head(10))
    
    print(f"\n8. Sample of merged data:")
    sample = pd.concat([X, y, meta["class_name"]], axis=1)
    print(sample.head(10))
    
    # Check if filenames actually match
    print(f"\n9. Filename check:")
    print(f"   First feature filename: {meta.iloc[0]['filename']}")
    print(f"   First label class_name: {meta.iloc[0]['class_name']}")

if __name__ == "__main__":
    main()