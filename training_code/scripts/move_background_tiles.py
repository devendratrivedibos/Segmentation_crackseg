import os
import cv2
import random
import shutil
from glob import glob

def is_all_black(mask_path):
    """Check if mask has only black pixels."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return cv2.countNonZero(mask) == 0

def process_masks(mask_dir, img_dir, save_mask_dir, save_img_dir, ext=".png"):
    os.makedirs(save_mask_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    mask_files = sorted(glob(os.path.join(mask_dir, f"*{ext}")))

    # sections to skip
    exclude_sections = [
        "AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1",
        "AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-2",
        "AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-3",
        "AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-4",
    ]
    mask_files = [mf for mf in mask_files if not any(ex in mf for ex in exclude_sections)]

    # Group by base filename before last "_"
    groups = {}
    for mf in mask_files:
        base = os.path.basename(mf)
        key = "_".join(base.split("_")[:-1])  # everything except last tile index
        groups.setdefault(key, []).append(mf)

    for key, tiles in groups.items():
        # separate black and non-black
        black_tiles = [t for t in tiles if is_all_black(t)]
        nonblack_tiles = [t for t in tiles if t not in black_tiles]

        if not black_tiles:
            continue  # nothing to move if no black tiles

        # move 90% of the black tiles only
        k = int(len(black_tiles) * 0.9)
        selected = random.sample(black_tiles, k)

        for s in selected:
            # move mask
            shutil.move(s, os.path.join(save_mask_dir, os.path.basename(s)))

            # move corresponding image
            img_name = os.path.basename(s).replace(ext, ".jpg")  # adjust if needed
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                shutil.move(img_path, os.path.join(save_img_dir, img_name))

        print(f"{key}: total={len(tiles)}, nonblack={len(nonblack_tiles)}, "
              f"black={len(black_tiles)}, moved={len(selected)} (90% of black)")

if __name__ == "__main__":
    mask_dir = r"G:\Devendra\ALL_MIX\TILE_HIGH_RES_MASKS"        # input mask folder
    img_dir = r"G:\Devendra\ALL_MIX\TILE_HIGH_RES_IMAGES"        # input image folder
    save_mask_dir = r"G:\Devendra\ALL_MIX\TILE_HIGH_RES_MASKS_BACKGROUND_70"  # destination for masks
    save_img_dir = r"G:\Devendra\ALL_MIX\TILE_HIGH_RES_IMAGES_BACKGROUND_70"  # destination for images

    process_masks(mask_dir, img_dir, save_mask_dir, save_img_dir, ext=".png")
