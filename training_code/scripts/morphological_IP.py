import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_bool
import os
from pathlib import Path

# --- Color map ---
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

# strictly only these will be skeletonized
SKELETONIZE_CLASSES = {
    (255, 0, 0),   # Alligator
    (0, 0, 255),   # Transverse Crack
    (0, 255, 0),   # Longitudinal Crack
}


def thicken_cracks_multicolor(mask_path, output_path, thickness=3):
    img = cv2.imread(mask_path)
    if img is None:
        print(f"[WARN] Could not read image: {mask_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # start with a copy of original so blobs stay intact
    output = img_rgb.copy()

    # process only cracks (thin line classes)
    for color in SKELETONIZE_CLASSES:
        color_mask = np.all(img_rgb == color, axis=-1)
        if np.any(color_mask):
            # remove original crack from output first
            output[color_mask] = (0, 0, 0)

            # skeletonize + thicken
            skeleton = skeletonize(img_as_bool(color_mask))
            skeleton_uint8 = (skeleton.astype(np.uint8)) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
            thick_crack = cv2.dilate(skeleton_uint8, kernel, iterations=1)

            # add back thickened crack in original color
            output[thick_crack > 0] = color

    # Save
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)


def process_directory(input_dir, output_dir, thickness=3):
    os.makedirs(output_dir, exist_ok=True)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    print(f"[INFO] Found {len(files)} images in {input_dir}")

    for i, file in enumerate(files, 1):
        out_path = output_dir / file.name
        print(f"[{i}/{len(files)}] Processing {file.name} -> {out_path.name}")
        thicken_cracks_multicolor(str(file), str(out_path), thickness=thickness)


# --- Example usage ---
input_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\ACCEPTED_MASKS"  # Folder with input crack masks
output_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\ACCEPTED_SKELETON"  # Folder to save processed masks
# process_directory(input_dir, output_dir, thickness=3)

input_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_results"  # Folder with input crack masks
output_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_results_skeleton"  # Folder to save processed masks
process_directory(input_dir, output_dir, thickness=3)

export DJANGO_DB = "postgres"
export DJANGO_DB_NAME = "postgres"
export DJANGO_DB_USER = "Admin  "
export DJANGO_DB_PASSWORD = "Bos@123"
export DJANGO_DB_HOST = "your_database_host"
export DJANGO_DB_PORT = "your_database_port"

import cv2
import numpy as np


def soft_iou(gt_path, pred_path, tolerance_px=3):
    """
    Calculates soft IoU for binary/multicolor crack masks, given pixel tolerance.
    Args:
        gt_path (str): Ground truth image file path
        pred_path (str): Predicted mask image file path
        tolerance_px (int): Pixel-wise match tolerance (3px typical)

    Returns:
        float: Soft IoU score
    """
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    print(inter/(union + 1e-7))
    # To binary: consider any non-black pixel as crack
    gt_bin = np.any(gt != [0, 0, 0], axis=-1).astype(np.uint8)
    pred_bin = np.any(pred != [0, 0, 0], axis=-1).astype(np.uint8)

    # Distance transform on predicted cracks (background = 1, crack = 0)
    dist = cv2.distanceTransform(1 - pred_bin, cv2.DIST_L2, 3)
    # Soft intersection: GT crack pixels within tolerance of prediction
    soft_intersection = np.sum((dist <= tolerance_px) & (gt_bin == 1))
    union = np.sum((gt_bin + pred_bin) > 0)
    return soft_intersection / (union + 1e-8)


def buffer_mask(mask_path, output_path, tolerance_px=3):
    """
    Expands crack regions in the mask by tolerance_px using distance transform.
    Non-black pixels are considered cracks. Saved mask has cracks buffered (thicker) for soft IoU.
    """
    img = cv2.imread(mask_path)
    crack_bin = np.any(img != [0, 0, 0], axis=-1).astype(np.uint8)

    # Distance transform: cracks = 1, bg = 0
    dist = cv2.distanceTransform(1 - crack_bin, cv2.DIST_L2, 3)
    buffered = (dist <= tolerance_px).astype(np.uint8) * 255

    # Save as binary (white crack, black background)
    cv2.imwrite(output_path, buffered)
    print("Saved buffered mask:", output_path)


score = soft_iou(
    gt_path= r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\ACCEPTED_MASKS\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0000753.png",
    # pred_path=r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_results\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0000753.png",
    pred_path=r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_results\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-2_IMG_0003035_og_buffered.png",
    tolerance_px=3)

print('Soft IoU:', score)


buffer_mask(
    mask_path=r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\ACCEPTED_MASKS\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0000753.png",
    output_path=r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\ACCEPTED_MASKS\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-2_IMG_0003035_og_buffered.jpg",
    tolerance_px=3
)