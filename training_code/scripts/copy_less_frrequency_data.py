import cv2
import os
import numpy as np
import shutil
import csv
import pandas as pd

# --- Color map (RGB) → ID ---
COLOR_MAP = {
    (0, 0, 0): 0,         # Black - Background
    (255, 0, 0): 1,       # Red - Alligator
    (0, 0, 255): 2,       # Blue - Transverse Crack
    (0, 255, 0): 3,       # Green - Longitudinal Crack
    (139, 69, 19): 4,     # Brown - Pothole
    (255, 165, 0): 5,     # Orange - Patches
    (255, 0, 255): 6,    # Violet - Multiple Crack
    (0, 255, 255): 7,     # Cyan - Spalling
    (0, 128, 0): 8,       # Dark Green - Corner Break
    (255, 100, 203): 9,   # Light Pink - Sealed Joint - T
    (199, 21, 133): 10,   # Dark Pink - Sealed Joint - L
    (128, 0, 128): 11,     # Purple - Punchout
    (112, 102, 255): 12,  #popout Grey
    (255, 255, 255): 13,  # White - Unclassified
    (255, 215, 0): 14,  # Gold - Cracking
}

# Target classes
target_classes = {2, 3, 4, 5, 6, 7, 8, 11, 12}  # Blue and Magenta

# Colors corresponding to target classes (RGB)
target_colors = [color for color, cls in COLOR_MAP.items() if cls in target_classes]

# Paths
mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\DATA\AnnotationMasks"
output_mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\DATA\AnnotationMasks"
image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\DATA\AnnotationImages"
output_image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\DATA\AnnotationImages"

os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)
matching_images = []


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

# rows = []
# for filename in matching_images:
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
            shutil.copy2(mask_src, os.path.join(output_mask_folder, f"{name}_copy3{mask_ext}"))
            shutil.copy2(mask_src, os.path.join(output_mask_folder, f"{name}_copy4{mask_ext}"))
            shutil.copy2(mask_src, os.path.join(output_mask_folder, f"{name}_copy5{mask_ext}"))

            # Copy image duplicates with original image extension
            shutil.copy2(image_src, os.path.join(output_image_folder, f"{name}_copy1{image_ext}"))
            shutil.copy2(image_src, os.path.join(output_image_folder, f"{name}_copy2{image_ext}"))
            shutil.copy2(image_src, os.path.join(output_image_folder, f"{name}_copy3{image_ext}"))
            shutil.copy2(image_src, os.path.join(output_image_folder, f"{name}_copy4{image_ext}"))
            shutil.copy2(image_src, os.path.join(output_image_folder, f"{name}_copy5{image_ext}"))
            print(f"✔ Copied duplicates for: {filename}")