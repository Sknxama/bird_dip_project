"""Clean and aggregate AVONET ecological data."""
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, OUTPUTS_DIR


def clean_avonet(input_name="avonet.csv", output_name="avonet_clean.csv"):
    """
    Clean AVONET: keep useful columns, average by species, handle missing values.
    
    Output columns:
        - species: scientific name
        - beak_length: culmen length (mm)
        - beak_width: width (mm)
        - beak_depth: depth (mm)
        - wing_length: wing length (mm)
        - tail_length: tail length (mm)
        - tarsus_length: tarsus length (mm)
    """
    path = os.path.join(DATA_DIR, input_name)
    
    # Load with latin-1 encoding (we know this works)
    df = pd.read_csv(path, encoding="latin-1")
    
    # Select useful columns
    cols = [
        "Species1_BirdLife",
        "Beak.Length_Culmen",
        "Beak.Width",
        "Beak.Depth",
        "Wing.Length",
        "Tail.Length",
        "Tarsus.Length",
    ]
    df = df[cols].copy()
    
    # Rename for simplicity
    df.columns = [
        "species",
        "beak_length",
        "beak_width",
        "beak_depth",
        "wing_length",
        "tail_length",
        "tarsus_length",
    ]
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Average by species (multiple specimens per species)
    df_clean = df.groupby("species").mean().reset_index()
    
    # Save
    output_path = os.path.join(OUTPUTS_DIR, output_name)
    df_clean.to_csv(output_path, index=False)
    
    print(f"AVONET cleaned successfully.")
    print(f"  Original rows: {len(df)}")
    print(f"  Unique species: {len(df_clean)}")
    print(f"  Saved to: {output_path}")
    
    return df_clean


if __name__ == "__main__":
    clean_avonet()