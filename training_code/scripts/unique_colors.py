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

# --- Define Folders ---
folders = [
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-4\Masks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3\Masks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2\Masks",
]

# folders = [
#     r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\04-09-2025\Masks",
#     r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\03-09-2025\Masks",
# ]

# Collect (folder, filename) tuples
image_files = []
for folder in folders:
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append((folder, f))

total_images = len(image_files)

# --- Color map (RGB) → ID ---
COLOR_MAP = {
    (0, 0, 0): 0,  # Black   - Background
    (255, 0, 0): 1,  # Red     - Alligator
    (0, 0, 255): 2,  # Blue    - Transverse Crack
    (0, 255, 0): 3,  # Green   - Longitudinal Crack
    (139, 69, 19): 4,  # Brown   - Pothole
    (255, 165, 0): 5,  # Orange  - Patches
    (128, 0, 128): 6,  # Purple  - Punchout
    (0, 255, 255): 7,  # Cyan    - Spalling
    (0, 128, 0): 8,  # Dark Green - Corner Break
    (255, 100, 203): 9,  # Light Pink - Sealed Joint - T
    (199, 21, 133): 10,  # Dark Pink  - Sealed Joint - L
    (255, 215, 0): 11,  # Gold       - Cracking
    (255, 255, 255): 12,  # White      - Unclassified
    (255, 0, 255): 13,  # Yellow   multiple crack
}

# --- Track stats ---
id_image_count = defaultdict(int)  # Count of images per class ID
invalid_color_files = {}  # Files containing unknown colors
id12_files = []
# --- Loop over images ---
iterator = tqdm(image_files, desc="Processing images") if use_tqdm else image_files

for idx, (folder, filename) in enumerate(iterator, start=1):
    path = os.path.join(folder, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"⚠️ Could not read: {path}")
        continue

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, image.shape[2])
    unique_colors_in_image = set(map(tuple, pixels))

    unknown_colors = set()
    known_ids_in_image = set()

    for color in unique_colors_in_image:
        if color in COLOR_MAP:
            known_ids_in_image.add(COLOR_MAP[color])
        else:
            unknown_colors.add(color)

    # Update counts
    for cid in known_ids_in_image:
        id_image_count[cid] += 1

    # Track invalid colors
    if unknown_colors:
        invalid_color_files[path] = unknown_colors

    # Track ID 12 images
    if 12 in known_ids_in_image:
        id12_files.append(path)

    # Print progress if tqdm not available
    if not use_tqdm and (idx % 100 == 0 or idx == total_images):
        print(f"Processed {idx}/{total_images} images")

# --- Summary Report ---
print("\n✔️ Image count per known class ID:")
for cid in sorted(id_image_count):
    print(f"ID {cid}: {id_image_count[cid]} image(s)")

if invalid_color_files:
    print("\n⚠️ Files with unknown colors:")
    for fname, colors in invalid_color_files.items():
        print(f"{fname} → {sorted(colors)}")
else:
    print("\n✅ All images use only valid colors.")

# --- Files containing ID=12 (White) ---
print("\n📂 Files containing ID=12 (Unclassified, White):")
for f in id12_files:
    print(f)
