import os
import cv2
import numpy as np
from collections import defaultdict

# Optional: tqdm for progress bar
try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    use_tqdm = False

# Folder path
mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_MASKS"

# Color map (RGB) → ID
COLOR_MAP = {
    (0, 0, 0): 0,           # Black   - Background     1139
    (255, 0, 0): 1,         # Red     - Alligator      700
    (0, 0, 255): 2,         # Blue    - Transverse Crack    90
    (0, 255, 0): 3,         # Green   - Longitudinal Crack    522
    (255, 0, 255): 4,       # Magenta - Multiple Crack        136
    (255, 204, 0): 5,       # Yellow  - Joint Seal            2
    (0, 42, 255): 6,         # Orange  - Pothole
    (255,255,255):7
}

# Track count of how many images each ID appears in
id_image_count = defaultdict(int)

# Track files with unknown colors
invalid_color_files = {}

# Collect image files
image_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
total_images = len(image_files)

# Loop over images
iterator = tqdm(image_files, desc="Processing images") if use_tqdm else image_files

for idx, filename in enumerate(iterator, start=1):
    path = os.path.join(mask_folder, filename)
    image = cv2.imread(path)  # BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
    if image is None:
        continue

    pixels = image.reshape(-1, image.shape[2])
    unique_colors_in_image = set(tuple(pixel) for pixel in pixels)

    unknown_colors = set()
    known_ids_in_image = set()

    for color in unique_colors_in_image:
        if color in COLOR_MAP:
            known_ids_in_image.add(COLOR_MAP[color])
        else:
            unknown_colors.add(color)

    for cid in known_ids_in_image:
        id_image_count[cid] += 1

    if unknown_colors:
        invalid_color_files[filename] = unknown_colors

    # Show progress if tqdm not used
    if not use_tqdm and (idx % 100 == 0 or idx == total_images):
        print(f"Processed {idx}/{total_images} images")

# --- Summary Report ---

print("\n✔️ Image count per known class ID:")
for cid in sorted(id_image_count):
    print(f"ID {cid}: {id_image_count[cid]} image(s)")

if invalid_color_files:
    print("\n⚠️ Files with unknown colors:")
    for fname, colors in invalid_color_files.items():
        print(f"{fname} ))#: {sorted(colors)}")

        # print(fname)
else:
    print("\n✅ All images use only valid colors.")
