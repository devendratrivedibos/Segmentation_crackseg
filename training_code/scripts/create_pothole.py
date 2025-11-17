import os
import cv2
import shutil
import numpy as np
from multiprocessing import Pool, cpu_count

# --- Paths ---
BASE_DIR = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5"
img_dir = os.path.join(BASE_DIR, "ACCEPTED_IMAGES")
mask_dir = os.path.join(BASE_DIR, "ACCEPTED_MASKS")
delete_mask_dir = os.path.join(BASE_DIR, "only_background_MASK")
delete_image_dir = os.path.join(BASE_DIR, "only_background_IMAGE")
pothole_mask_dir = os.path.join(BASE_DIR, "pothole_MASK")
pothole_image_dir = os.path.join(BASE_DIR, "pothole_IMAGE")

os.makedirs(delete_mask_dir, exist_ok=True)
os.makedirs(delete_image_dir, exist_ok=True)
os.makedirs(pothole_mask_dir, exist_ok=True)
os.makedirs(pothole_image_dir, exist_ok=True)

# --- Color map (RGB) → ID ---
COLOR_MAP = {
    (0, 0, 0): 0,         # Background
    (255, 0, 0): 1,       # Alligator
    (0, 0, 255): 2,       # Transverse Crack
    (0, 255, 0): 3,       # Longitudinal Crack
    (139, 69, 19): 4,     # Pothole
    (255, 165, 0): 5,     # Patches
    (255, 0, 255): 6,     # Multiple Crack
    (0, 255, 255): 7,     # Spalling
    (0, 128, 0): 8,       # Corner Break
    (255, 100, 203): 9,   # Sealed Joint - T
    (199, 21, 133): 10,   # Sealed Joint - L
    (128, 0, 128): 11,    # Punchout
    (112, 102, 255): 12,  # Popout Grey
    (255, 255, 255): 13,  # Unclassified
    (255, 215, 0): 14,    # Cracking
}

COLOR_TO_ID = {k: v for k, v in COLOR_MAP.items()}
DELETE_SETS = [{0, 4}, {1, 4}]  # Optional; your current logic


def find_image(mask_name):
    """Return image path with same name but png or jpg extension."""
    base_name = os.path.splitext(mask_name)[0]
    for ext in (".png", ".jpg"):
        img_path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def process_mask(mask_name: str):
    """Move or copy mask & corresponding image based on content."""
    if not mask_name.lower().endswith(".png"):
        return None

    mask_path = os.path.join(mask_dir, mask_name)
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path)
    if mask is None:
        return None

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    unique_colors = {tuple(c) for c in mask_rgb.reshape(-1, 3)}
    unique_ids = {COLOR_TO_ID.get(c, -1) for c in unique_colors if c in COLOR_TO_ID}

    img_path = find_image(mask_name)

    # # --- Case 1: Move unwanted sets ---
    # if unique_ids in DELETE_SETS:
    #     try:
    #         shutil.move(mask_path, os.path.join(delete_mask_dir, mask_name))
    #         if img_path:
    #             shutil.move(img_path, os.path.join(delete_image_dir, os.path.basename(img_path)))
    #         return f"Deleted: {mask_name}"
    #     except Exception as e:
    #         return f"Error moving {mask_name}: {e}"

    # --- Case 2: Copy pothole images ---
    if 4 in unique_ids:
        try:
            shutil.copy(mask_path, os.path.join(pothole_mask_dir, mask_name))
            if img_path:
                shutil.copy(img_path, os.path.join(pothole_image_dir, os.path.basename(img_path)))
            return f"Pothole: {mask_name}"
        except Exception as e:
            return f"Error copying {mask_name}: {e}"

    return None


if __name__ == "__main__":
    all_masks = os.listdir(mask_dir)

    with Pool(processes=cpu_count()) as pool:
        results = list(pool.map(process_mask, all_masks))

    moved = [r for r in results if r and "Deleted" in r]
    copied = [r for r in results if r and "Pothole" in r]
    print(f"✅ Done.\nMoved {len(moved)} unwanted mask-image pairs.\nCopied {len(copied)} pothole pairs.")
