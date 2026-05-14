import os
import cv2
import random
import shutil
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================
# CONFIG
# =====================
REMOVE_PERCENTAGE = 0.6  # remove 60% of black tiles
MAX_WORKERS = 6  # adjust CPU usage

mask_ext = ".png"

mask_dir = r"Z:\Devendra\CONCRETE\TILE\TILE_HIGH_RES_MASKS"
img_dir = r"Z:\Devendra\CONCRETE\TILE\TILE_HIGH_RES_IMAGES"

save_mask_dir = r"Z:\Devendra\CONCRETE\TILE\MASKS"
save_img_dir = r"Z:\Devendra\CONCRETE\TILE\IMAGES"

os.makedirs(save_mask_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)


# =====================
# HELPERS
# =====================
def is_all_black(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return cv2.countNonZero(mask) == 0


def find_image_file(base_name):
    for ext in [".jpg", ".png", ".jpeg"]:

        img_path = os.path.join(img_dir, base_name + ext)

        if os.path.exists(img_path):
            return img_path

    return None


def process_group(item):
    key, tiles = item

    black_tiles = []

    for t in tiles:
        if is_all_black(t):
            black_tiles.append(t)

    if len(black_tiles) == 0:
        return f"{key} | no black tiles"

    remove_count = int(len(black_tiles) * REMOVE_PERCENTAGE)

    if remove_count == 0:
        return f"{key} | black tiles too few"

    selected = random.sample(black_tiles, remove_count)

    for mask_path in selected:

        base_name = os.path.splitext(os.path.basename(mask_path))[0]

        # move mask
        shutil.move(
            mask_path,
            os.path.join(save_mask_dir, os.path.basename(mask_path))
        )

        # move corresponding image
        img_path = find_image_file(base_name)

        if img_path:
            shutil.move(
                img_path,
                os.path.join(save_img_dir, os.path.basename(img_path))
            )

    return (
        f"{key} | total={len(tiles)} | "
        f"black={len(black_tiles)} | removed={len(selected)}"
    )


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    mask_files = sorted(glob(os.path.join(mask_dir, f"*{mask_ext}")))

    print("Total masks:", len(mask_files))

    # group tiles by parent image
    groups = {}

    for mf in mask_files:
        base = os.path.basename(mf)
        key = "_".join(base.split("_")[:-1])
        groups.setdefault(key, []).append(mf)
    print("Total groups:", len(groups))
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_group, item)
            for item in groups.items()
        ]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(result)
            print(f"Progress: {completed}/{len(groups)}")
    print("\n✅ Multiprocessing removal complete")