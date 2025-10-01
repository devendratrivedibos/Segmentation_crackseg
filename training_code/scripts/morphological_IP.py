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
SKELETONIZE_CLASSES = {1,2,3}


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


# score = soft_iou(
#     gt_path= r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\ACCEPTED_MASKS\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0000753.png",
#     # pred_path=r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_results\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0000753.png",
#     pred_path=r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_results\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-2_IMG_0003035_og_buffered.png",
#     tolerance_px=3)

# print('Soft IoU:', score)
# process_directory(input_dir, output_dir, thickness=3)


def calculate_skeleton_length(mask_rgb, skeletonize_cracks=True, measure_length=True):
    """
    Process an RGB mask:
      - Skeletonize only cracks
      - Leave blobs (potholes/patches) untouched
      - Optionally calculate crack length

    Args:
        mask_rgb (np.ndarray): HxWx3 RGB mask
        skeletonize_cracks (bool): Whether to skeletonize cracks
        measure_length (bool): Whether to calculate crack length

    Returns:
        processed_mask (np.ndarray): HxWx3 RGB mask (skeletonized cracks)
        crack_lengths (dict): length of cracks per class
    """
    h, w, _ = mask_rgb.shape
    processed_mask = np.zeros_like(mask_rgb)
    crack_lengths = {}

    for color, (cls_id, cls_name) in COLOR_MAP.items():
        # Binary mask for this class
        binary = np.all(mask_rgb == color, axis=-1).astype(np.uint8)

        if binary.sum() == 0:
            continue

        if cls_id in SKELETONIZE_CLASSES and skeletonize_cracks:
            # Skeletonize cracks
            skeleton = skeletonize(binary > 0).astype(np.uint8)

            # Add to output in original color
            coords = np.where(skeleton > 0)
            processed_mask[coords] = color

            if measure_length:
                # crack length ≈ number of skeleton pixels
                crack_lengths[cls_name] = int(skeleton.sum())

        else:
            # Keep blobs / non-cracks unchanged
            coords = np.where(binary > 0)
            processed_mask[coords] = color

            if measure_length and cls_id not in SKELETONIZE_CLASSES:
                crack_lengths[cls_name] = int(binary.sum())  # area, not length

    print(f"Processed mask size: {h}x{w}, Crack lengths: {crack_lengths}")
    return processed_mask, crack_lengths

img = cv2.imread(r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_results\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0004853.png")
a,b = calculate_skeleton_length(img)

img = cv2.imread(r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\ACCEPTED_MASKS\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0004853.png")
a,b = calculate_skeleton_length(img)

