import os
import cv2
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# --- Define Folders ---
# Concreete Day
folders = [
    r"Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-1\AnnotationMasks",
    r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\DayMasks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-7\AnnotationMasks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-6\AnnotationMasks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-5\AnnotationMasks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-4\AnnotationMasks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3\AnnotationMasks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2\AnnotationMasks",
]

##ASPHALT DAY
folders = [
    r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-1\AnnotationMasks",
    r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-2\AnnotationMasks"
]

## CONCRETE DAY
# folders = [r"Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-3\AnnotationMasks",
#             r"Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-4\AnnotationMasks",
# r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\AnnotationMasksNIGHT"
# ]

# folders=[
#         # r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\AnnotationMasks",
#         r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATA_V2\AnnotationMasks"
#         ]

# Collect (folder, filename) tuples
image_files = []
for folder in folders:
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append((folder, f))

total_images = len(image_files)

# --- Color map (RGB) → ID ---
COLOR_MAP = {
    (0, 0, 0): 0,         # Black - Background
    (255, 0, 0): 1,       # Red - Alligator
    (0, 0, 255): 2,       # Blue - Transverse Crack
    (0, 255, 0): 3,       # Green - Longitudinal Crack
    (139, 69, 19): 4,     # Brown - Pothole
    (255, 165, 0): 5,     # Orange - Patches
    (128, 0, 128): 6,     # Purple - Punchout
    (0, 255, 255): 7,     # Cyan - Spalling
    (0, 128, 0): 8,       # Dark Green - Corner Break
    (255, 100, 203): 9,   # Light Pink - Sealed Joint - T
    (199, 21, 133): 10,   # Dark Pink - Sealed Joint - L
    (255, 215, 0): 11,    # Gold - Cracking
    (255, 255, 255): 12,  # White - Unclassified
    (255, 0, 255): 13,    # Violet - Multiple Crack
    (112, 102, 255): 14,  # Grey
}

# --- Worker function ---
def process_image(COLOR_MAP, item):
    folder, filename = item
    path = os.path.join(folder, filename)

    image = cv2.imread(path)
    if image is None:
        return {"path": path, "error": True}

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, image.shape[2])
    unique_colors = set(map(tuple, pixels))

    unknown_colors = set()
    known_ids = set()

    for color in unique_colors:
        if color in COLOR_MAP:
            known_ids.add(COLOR_MAP[color])
        else:
            unknown_colors.add(color)

    return {
        "path": path,
        "error": False,
        "known_ids": known_ids,
        "unknown_colors": unknown_colors,
    }

# --- Main multiprocessing logic ---
if __name__ == "__main__":
    print(f"🔍 Processing {total_images} images using {cpu_count()} CPUs...")

    results = []
    with Pool(processes=cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(partial(process_image, COLOR_MAP), image_files),
                        total=total_images, desc="Processing images"):
            results.append(res)

    # --- Aggregate results ---
    id_image_count = defaultdict(int)
    invalid_color_files = {}
    id12_files = []

    for res in results:
        if res["error"]:
            print(f"⚠️ Could not read: {res['path']}")
            continue

        for cid in res["known_ids"]:
            id_image_count[cid] += 1

        if res["unknown_colors"]:
            invalid_color_files[res["path"]] = res["unknown_colors"]

        if 12 in res["known_ids"]:
            id12_files.append(res["path"])

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

    print("\n📂 Files containing ID=12 (Unclassified, White):")
    for f in id12_files:
        print(f)
