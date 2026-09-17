"""Map CUB-200-2011 bird classes to AVONET ecological traits."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import difflib
from config import OUTPUTS_DIR, ANNOTATIONS_DIR


def load_class_names():
    """Load CUB class names from classes.txt."""
    path = os.path.join(ANNOTATIONS_DIR, "classes.txt")
    df = pd.read_csv(path, sep=" ", header=None, names=["class_id", "class_name"])
    return df


def load_avonet_clean():
    """Load cleaned AVONET data."""
    path = os.path.join(OUTPUTS_DIR, "avonet_clean.csv")
    return pd.read_csv(path)


def get_best_match(class_name, avonet_df):
    """
    Find best matching AVONET species using fuzzy matching.
    Strategy: compare CUB common name against AVONET scientific names.
    """
    # Clean CUB name: "001.Black_footed_Albatross" -> "Black footed Albatross"
    clean_name = class_name.split(".")[-1].replace("_", " ").lower()
    
    # Get all AVONET scientific names
    avonet_names = avonet_df["species"].tolist()
    
    # Find best match using difflib
    matches = difflib.get_close_matches(clean_name, avonet_names, n=1, cutoff=0.4)
    
    if matches:
        best_match = matches[0]
        result = avonet_df[avonet_df["species"] == best_match].iloc[0].to_dict()
        result["source"] = "AVONET matched"
        return result
    
    # Fallback: global averages
    result = avonet_df.mean(numeric_only=True).to_dict()
    result["species"] = "Global average"
    result["source"] = "AVONET global average"
    return result


def build_eco_lookup_table(output_name="eco_lookup.csv"):
    """
    Build a lookup table: CUB class_id -> AVONET ecological traits.
    """
    cub_classes = load_class_names()
    avonet = load_avonet_clean()
    
    records = []
    matched_count = 0
    
    for _, row in cub_classes.iterrows():
        traits = get_best_match(row["class_name"], avonet)
        traits["class_id"] = row["class_id"]
        traits["class_name_cub"] = row["class_name"]
        if traits["source"] == "AVONET matched":
            matched_count += 1
        records.append(traits)
    
    df = pd.DataFrame(records)
    output_path = os.path.join(OUTPUTS_DIR, output_name)
    df.to_csv(output_path, index=False)
    
    print(f"Eco lookup table built: {len(df)} classes mapped.")
    print(f"  Direct matches: {matched_count}")
    print(f"  Global averages: {len(df) - matched_count}")
    print(f"Saved to: {output_path}")
    print(f"\nSample (first 5 rows):")
    print(df[["class_name_cub", "species", "beak_length", "wing_length", "source"]].head())
    
    return df


if __name__ == "__main__":
    build_eco_lookup_table()