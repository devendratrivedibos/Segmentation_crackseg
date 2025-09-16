import os
import cv2
import shutil
import numpy as np
from multiprocessing import Pool, cpu_count

# --- Paths ---
BASE_DIR = r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-7"
mask_dir = os.path.join(BASE_DIR, "AnnotationMasks")
img_dir = os.path.join(BASE_DIR, "AnnotationImages")

delete_mask_dir = os.path.join(BASE_DIR, "only_JS_MASK")
delete_image_dir = os.path.join(BASE_DIR, "only_JS_IMAGE")

os.makedirs(delete_mask_dir, exist_ok=True)
os.makedirs(delete_image_dir, exist_ok=True)

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

COLOR_TO_ID = {k: v for k, v in COLOR_MAP.items()}

DELETE_SETS = [{0, 9}, {0, 10}, {0, 9, 10}]


def find_image(mask_name):
    """Return image path with same name but png or jpg extension."""
    base_name = os.path.splitext(mask_name)[0]
    for ext in (".png", ".jpg"):
        img_path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def process_mask(mask_name: str):
    """Move mask & corresponding image if only unwanted IDs."""
    if not mask_name.lower().endswith(".png"):
        return None

    mask_path = os.path.join(mask_dir, mask_name)
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path)
    if mask is None:
        return None

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    pixels = mask_rgb.reshape(-1, 3)
    unique_colors = {tuple(color) for color in pixels}
    unique_ids = {COLOR_TO_ID.get(c, -1) for c in unique_colors if c in COLOR_TO_ID}

    if unique_ids in DELETE_SETS:
        try:
            # Move mask
            shutil.move(mask_path, os.path.join(delete_mask_dir, mask_name))

            # Move image (png or jpg)
            img_path = find_image(mask_name)
            if img_path:
                shutil.move(img_path, os.path.join(delete_image_dir, os.path.basename(img_path)))

            return mask_name
        except Exception as e:
            return f"Error moving {mask_name}: {e}"
    return None


if __name__ == "__main__":
    all_masks = os.listdir(mask_dir)

    with Pool(processes=cpu_count()) as pool:
        results = list(pool.map(process_mask, all_masks))

    moved = [r for r in results if r]
    print(f"✅ Done. Moved {len(moved)} mask-image pairs into DELETE_MASK & DELETE_IMAGE.")
