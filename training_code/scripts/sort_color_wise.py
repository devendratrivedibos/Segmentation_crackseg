import cv2
import os
import numpy as np
import shutil
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
target_classes = {2, 3, 4, 5, 12}  # Blue and Magenta
target_colors = {cls: color for color, cls in COLOR_MAP.items() if cls in target_classes}

# Paths
base_output = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2"
mask_folder = os.path.join(base_output, "DATASET_MASKS_")
image_folder = os.path.join(base_output, "DATASET_IMAGES_")

# Separate folders for each class
output_folders = {}
for cls in target_classes:
    output_folders[cls] = {
        "mask": os.path.join(base_output, f"DATASET_MASKS_CLASS_{cls}"),
        "image": os.path.join(base_output, f"DATASET_IMAGES_CLASS_{cls}")
    }
    os.makedirs(output_folders[cls]["mask"], exist_ok=True)
    os.makedirs(output_folders[cls]["image"], exist_ok=True)


def find_image_file(image_folder, base_name):
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = os.path.join(image_folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


rows = []
matching_images = []

# Find masks with target colors
for filename in os.listdir(mask_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Check which target class is present
        for cls, color in target_colors.items():
            if np.any(np.all(mask == color, axis=-1)):
                matching_images.append((filename, cls))
                # break  # assign to first found class only

# Copy and log
for filename, cls in matching_images:
    name, _ = os.path.splitext(filename)
    mask_src = os.path.join(mask_folder, filename)
    image_src = find_image_file(image_folder, name)

    if image_src is None:
        print(f"⚠ Image for {filename} not found, skipping.")
        continue

    _, mask_ext = os.path.splitext(filename)
    _, image_ext = os.path.splitext(image_src)

    # Output dirs by class
    mask_out = os.path.join(output_folders[cls]["mask"], f"{name}_copy1{mask_ext}")
    img_out = os.path.join(output_folders[cls]["image"], f"{name}_copy1{image_ext}")

    # Copy
    shutil.copy2(mask_src, mask_out)
    shutil.copy2(image_src, img_out)

    # Log row
    rows.append([cls, image_src, mask_src])

# Save CSV
df_new = pd.DataFrame(rows, columns=["class", "image_path", "mask_path"])
csv_path = os.path.join(base_output, "data.csv")
df_new.to_csv(csv_path, mode="a", index=False, header=False)

print(f"✅ Done! {len(matching_images)} masks & images copied into class-specific folders.")
