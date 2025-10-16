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
    # r"Y:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-1\ACCEPTED_MASKS",
    # r"Y:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-2\ACCEPTED_MASKS",
    # r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\ACCEPTED_MASKS",
    # r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-2\ACCEPTED_MASKS",
    # r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-3\ACCEPTED_MASKS",
    # r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-4\ACCEPTED_MASKS",
    r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5\ACCEPTED_MASKS",
    # r"T:\SHINGOTE-KOLHAR_2025-09-23_14-06-00\SECTION-1\process_distress_seg_masks",
    # r"T:\SHINGOTE-KOLHAR_2025-09-23_14-06-00\SECTION-2\process_distress_seg_masks"

]
# # CONCRETE DAY
# folders = [r"Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-3\AnnotationMasks",
# r"Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-4\AnnotationMasks",
# r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\AnnotationMasksNIGHT"
# ]

image_files = []
for folder in folders:
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and ("AMRAVTI" in f):
            image_files.append((folder, f))

total_images = len(image_files)

# --- Color map (RGB) → (ID, Name) ---
COLOR_MAP = {
    (0, 0, 0): (0, "Background"),
    (255, 0, 0): (1, "Alligator"),
    (0, 0, 255): (2, "Transverse Crack"),
    (0, 255, 0): (3, "Longitudinal Crack"),
    (139, 69, 19): (4, "Pothole"),
    (255, 165, 0): (5, "Patches"),
    (255, 0, 255): (6, "Multiple Crack"),
    (0, 255, 255): (7, "Spalling"),
    (0, 128, 0): (8, "Corner Break"),
    (255, 100, 203): (9, "Sealed Joint Transverse"),
    (199, 21, 133): (10, "Sealed Joint Longitudinal"),
    (128, 0, 128): (11, "Punchout"),
    (112, 102, 255): (12, "Popout"),
    (255, 255, 255): (13, "Unclassified"),
    (255, 215, 0): (14, "Cracking"),
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
            cid, cname = COLOR_MAP[color]
            known_ids.add(cid)
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
    name_image_count = defaultdict(int)
    invalid_color_files = {}
    unclassified = []

    for res in results:
        if res["error"]:
            print(f"⚠️ Could not read: {res['path']}")
            continue

        for cid in res["known_ids"]:
            # Increment both ID and name counts
            cname = [name for (rgb, (idx, name)) in COLOR_MAP.items() if idx == cid][0]
            id_image_count[cid] += 1
            name_image_count[cname] += 1

        if res["unknown_colors"]:
            invalid_color_files[res["path"]] = res["unknown_colors"]

        if 13 in res["known_ids"]:  # Unclassified
            unclassified.append(res["path"])

    # --- Summary Report ---
    print("\n✔️ Image count per known class ID:")
    for cid in sorted(id_image_count):
        cname = [name for (rgb, (idx, name)) in COLOR_MAP.items() if idx == cid][0]
        print(f"ID-{cid:2d} ({cname}): {id_image_count[cid]} image(s)")

    print("\n✔️ Image count per class Name:")
    for cname in sorted(name_image_count):
        print(f"{cname:20s}: {name_image_count[cname]} image(s)")

    if invalid_color_files:
        print("\n⚠️ Files with unknown colors:")
        for fname, colors in invalid_color_files.items():
            print(f"{fname} → {sorted(colors)}")
    else:
        print("\n✅ All images use only valid colors.")

    print("\n📂 Files containing ID=13 (Unclassified, White):")
    for f in unclassified:
        print(f)
