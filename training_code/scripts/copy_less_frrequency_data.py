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
    (255, 0, 255): 13,  # Violet - Multiple Crack
    (112, 102, 255): 14,  # Grey
}

# Target classes
target_classes = {13}  # Blue and Magenta

# Colors corresponding to target classes (RGB)
target_colors = [color for color, cls in COLOR_MAP.items() if cls in target_classes]

# Paths
mask_folder = r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-5\AnnotationMasks"
output_mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\AnnotationMasks_COPY"
image_folder = r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-5\AnnotationImages"
output_image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\AnnotationImages_COPY"
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

rows = []
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
