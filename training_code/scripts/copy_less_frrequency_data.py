import cv2
import os
import numpy as np
import shutil
import csv
import pandas as pd

# Color map
COLOR_MAP = {
    (0, 0, 0): 0,  # Black   - Background     1139
    (255, 0, 0): 1,  # Red     - Alligator      700
    (0, 0, 255): 2,  # Blue    - Transverse Crack    90
    (0, 255, 0): 3,  # Green   - Longitudinal Crack    522
    (139, 69, 19): 4,  # Brown    -  POTHOLE
    (255, 165, 0): 5,  # Orange   - PATCHES
    (128, 0, 128): 6,  # Purple   - punchout
    (0, 255, 255): 7,  # Cyan     -spalling
    (0, 128, 0): 8,  # Dark Green   - COrner Break
    (255, 100, 203): 9,  # Light Pink    - SEALED JOINT - T
    (199, 21, 133): 10,  # Dark Pink  - SEALED JOINT - L
    (255, 215, 0): 11,  # Gold    CRACKING
    (255, 255, 255): 12,  # WHITE  UNCLASSIFIED
}

# Target classes
target_classes = {2, 4, 5, 12}  # Blue and Magenta

# Colors corresponding to target classes (RGB)
target_colors = [color for color, cls in COLOR_MAP.items() if cls in target_classes]

# Paths
mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATA_TRAIN\DATASET_MASKS_"
output_mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATA_TRAIN\DATASET_MASKS_COPY"
image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATA_TRAIN\DATASET_IMAGES_"
output_image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATA_TRAIN\DATASET_IMAGES_COPY"
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)
matching_images = []

"""
for filename in os.listdir(mask_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(mask_folder, filename)

        # Read as RGB
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Check if any target color is present
        found = False
        for color in target_colors:
            if np.any(np.all(img == color, axis=-1)):
                found = True
                break

        if found:
            matching_images.append(filename)

print("Images matching classes 2 or 4:")
for name in matching_images:
    print(name)

print(f"Total: {len(matching_images)} images found.")
"""


def find_image_file(image_folder, base_name):
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = os.path.join(image_folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None

# Find matching masks
for filename in os.listdir(mask_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask_path = os.path.join(mask_folder, filename)

        # Read mask in RGB
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Check for target colors
        if any(np.any(np.all(mask == color, axis=-1)) for color in target_colors):
            matching_images.append(filename)

rows= []
for filename in matching_images:
    name, _ = os.path.splitext(filename)

    mask_src = os.path.join(mask_folder, filename)
    image_src = find_image_file(image_folder, name)

    if image_src is None:
        print(f"⚠ Image for {filename} not found, skipping.")
        continue

    _, mask_ext = os.path.splitext(filename)
    _, image_ext = os.path.splitext(image_src)

    # Copy mask duplicates
    shutil.copy2(mask_src, os.path.join(output_mask_folder, f"{name}_copy1{mask_ext}"))
    shutil.copy2(mask_src, os.path.join(output_mask_folder, f"{name}_copy2{mask_ext}"))

    # Copy image duplicates with original image extension
    shutil.copy2(image_src, os.path.join(output_image_folder, f"{name}_copy1{image_ext}"))
    shutil.copy2(image_src, os.path.join(output_image_folder, f"{name}_copy2{image_ext}"))
    rows.append([image_src, mask_src])

# # ✅ Save all at once
# df_new = pd.DataFrame(rows, columns=["image_path", "mask_path"])
#
# df_new.to_csv(r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\data.csv", mode="a", index=False, header=False)
# print(f"✅ Done! {len(matching_images)} masks and images copied (with duplicates).")
#
# print(f"✅ Done! {len(matching_images)} matching masks found.")
# print(f"Copied masks to: {output_mask_folder}")
# print(f"Copied images to: {output_image_folder}")
