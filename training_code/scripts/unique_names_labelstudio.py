import os
import pandas as pd

# Path to your folder
main_folder = r"F:\TRAINING_DATA\08_aug\segmentation_with_or_without_cracks_08_aug\ANNOTATION 08.08.2025"

# Keywords to search
keywords = ["alligator", "longitudinal", "transverse", "joint sealed transverse", "joint sealed longitudinal",
            "corner break", "pothole", "patch", "punchout", "spalling"]

# Allowed image extensions
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Initialize total counts
total_counts = {k: 0 for k in keywords}

# Optional: per-folder or per-file breakdown
per_file_counts = {}   # {filepath: {keyword: count(0/1)}}
per_folder_counts = {} # {folder_path: {keyword: total}}

for root, dirs, files in os.walk(main_folder):
    # Ensure folder accumulator exists
    if root not in per_folder_counts:
        per_folder_counts[root] = {k: 0 for k in keywords}

    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext in image_exts:
            filepath = os.path.join(root, fname)
            name_lower = fname.lower()

            # Count keyword presence per file (0/1 since it's filename-based)
            per_file_counts[filepath] = {k: (1 if k in name_lower else 0) for k in keywords}

            # Update totals
            for k in keywords:
                total_counts[k] += per_file_counts[filepath][k]
                per_folder_counts[root][k] += per_file_counts[filepath][k]

# Print totals
print("Total counts across all image filenames:")
for k, v in total_counts.items():
    print(f"{k}: {v}")

# # Optional: print per-folder summary
# print("\nPer-folder counts:")
# for folder, counts in per_folder_counts.items():
#     # Skip folders with zero images matched at all to keep output clean
#     if sum(counts.values()) == 0:
#         continue
#     print(f"\n{folder}")
#     for k, v in counts.items():
#         print(f"  {k}: {v}")

# # Optional: print per-file matches (only files that matched at least one keyword)
# print("\nPer-file matches:")
# for filepath, counts in per_file_counts.items():
#     if sum(counts.values()) > 0:
#         tags = [k for k, v in counts.items() if v]
#         print(f"{filepath}: {', '.join(tags)}")
