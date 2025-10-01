import os
import cv2
import random
import shutil
from glob import glob

def is_black(mask_path):
    """Check if mask is completely black."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return cv2.countNonZero(mask) == 0

def process_tiles(mask_dir, img_dir, save_mask_dir, save_img_dir, mask_ext=".png", img_ext=".jpg"):
    os.makedirs(save_mask_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    mask_files = sorted(glob(os.path.join(mask_dir, f"*{mask_ext}")))

    # Group tiles by base filename before last "_"
    groups = {}
    for mf in mask_files:
        base = os.path.basename(mf)
        key = "_".join(base.split("_")[:-1])  # group by big image
        groups.setdefault(key, []).append(mf)

    for key, tiles in groups.items():
        black_tiles = []
        non_black_tiles = []

        for t in tiles:
            if is_black(t):
                black_tiles.append(t)
            else:
                non_black_tiles.append(t)

        if not black_tiles:
            continue  # skip if no black tiles

        # select 70% of black tiles to move
        k = int(len(black_tiles) * 0.7)
        if k == 0 and len(black_tiles) > 0:
            k = 1  # ensure at least 1 is moved if some black exist
        selected = random.sample(black_tiles, k)

        for s in selected:
            # move mask
            shutil.move(s, os.path.join(save_mask_dir, os.path.basename(s)))

            # move corresponding image
            img_name = os.path.basename(s).replace(mask_ext, img_ext)
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                shutil.move(img_path, os.path.join(save_img_dir, img_name))

        print(f"{key}: {len(non_black_tiles)} non-black, {len(black_tiles)} black → moved {len(selected)}")


if __name__ == "__main__":
    mask_dir = r"G:\Devendra\ALL_MIX\TILE_HIGH_RES_MASKS"        # input mask folder
    img_dir = r"G:\Devendra\ALL_MIX\TILE_HIGH_RES_IMAGES"        # input image folder
    save_mask_dir = r"G:\Devendra\ALL_MIX\TILE_HIGH_RES_MASKS_BACKGROUND_70"  # destination for masks
    save_img_dir = r"G:\Devendra\ALL_MIX\TILE_HIGH_RES_IMAGES_BACKGROUND_70"    # destination for images
    process_tiles(mask_dir, img_dir, save_mask_dir, save_img_dir,
                  mask_ext=".png", img_ext=".jpg")

