import os

import pandas as pd

# Base path
base_dir = r"T:\SHINGOTE-KOLHAR_2025-09-23_14-06-00\SECTION-3"

# Paths to the CSV files
cracks_csv = os.path.join(base_dir, "process_distress", "cracks_predictions.csv")
potholes_csv = os.path.join(base_dir, "reportNew", "csv_reports", "potholes.csv")
patches_csv = os.path.join(base_dir, "reportNew", "csv_reports", "patches.csv")

# Collect CSV paths
csv_files = [cracks_csv, potholes_csv, patches_csv]
csv_files = {
    "cracks": cracks_csv,
    "potholes": potholes_csv,
    "patches": patches_csv
}
# Read and collect all region indices
all_region_indices = []

for name, csv_file in csv_files.items():
    try:
        df = pd.read_csv(csv_file)

        # Normalize region index column name
        col = "Region_index" #if "region index" in df.columns else "region_index"

        if name == "cracks":
            # take all
            if col in df.columns:
                all_region_indices.extend(df[col].dropna().tolist())

        elif name == "patches":
            if col in df.columns and "Number of patches" in df.columns:
                filtered = df[df["Number of patches"] > 0]
                all_region_indices.extend(filtered[col].dropna().tolist())

        elif name == "potholes":
            if col in df.columns and "Number of Potholes" in df.columns:
                filtered = df[df["Number of Potholes"] > 0]
                all_region_indices.extend(filtered[col].dropna().tolist())

        else:
            print(f"⚠️ Skipping {csv_file} (not cracks/patches/potholes)")

    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# Remove duplicates and sort
all_region_indices = sorted(set(all_region_indices))

print("✅ Total Region Indices found:", len(all_region_indices))
print(all_region_indices)
print(len(all_region_indices))
