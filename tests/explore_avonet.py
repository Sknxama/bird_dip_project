"""Explore AVONET structure automatically."""

import pandas as pd
import os
from config import DATA_DIR

def main():
    path = os.path.join(DATA_DIR, "avonet.csv")
    
    # Try different encodings
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    df = None
    
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded successfully with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        print("Could not load file with any encoding.")
        return
    
    print("=" * 60)
    print("AVONET EXPLORATION REPORT")
    print("=" * 60)
    
    print(f"\n1. Total rows (species): {len(df)}")
    print(f"2. Total columns: {len(df.columns)}")
    
    print(f"\n3. All column names:")
    for i, col in enumerate(df.columns):
        print(f"   {i}: {col}")
    
    # Search for mass/weight columns
    mass_keywords = ['mass', 'weight', 'body', 'corp', 'gravity']
    mass_cols = [c for c in df.columns if any(k in c.lower() for k in mass_keywords)]
    print(f"\n4. Columns related to mass/weight: {mass_cols if mass_cols else 'NONE FOUND'}")
    
    # Search for wing/span columns
    wing_keywords = ['wing', 'span', 'envergadura']
    wing_cols = [c for c in df.columns if any(k in c.lower() for k in wing_keywords)]
    print(f"5. Columns related to wings: {wing_cols}")
    
    # Show first 3 rows of key columns
    key_cols = ['Species1_BirdLife'] + mass_cols + wing_cols + ['Beak.Length_Culmen', 'Tarsus.Length']
    key_cols = [c for c in key_cols if c in df.columns]
    
    print(f"\n6. Sample data (first 3 rows):")
    print(df[key_cols].head(3).to_string())
    
    # Check for missing values in key columns
    print(f"\n7. Missing values in key columns:")
    for col in key_cols:
        missing = df[col].isna().sum()
        print(f"   {col}: {missing} missing")

if __name__ == "__main__":
    main()