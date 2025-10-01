import os
import shutil
from multiprocessing import Pool, cpu_count

# --- CONFIG ---
src_images_dir = r"Z:\Devendra\TILED\TILE_HIGH_RES_IMAGES"   # OneDrive images folder
src_masks_dir = r"Z:\Devendra\TILED\TILE_HIGH_RES_MASKS"    # OneDrive masks folder

dst_images_dir = r"V:\Devendra\TILE_HIGH_RES_IMAGES"   # destination images folder
dst_masks_dir = r"V:\Devendra\TILE_HIGH_RES_MASKS"    # destination masks folder

os.makedirs(dst_images_dir, exist_ok=True)
os.makedirs(dst_masks_dir, exist_ok=True)


def copy_file(filename):
    """Copy image and mask pair if they exist"""
    try:
        # source paths
        src_img = os.path.join(src_images_dir, filename)
        src_mask = os.path.join(src_masks_dir, os.path.splitext(filename)[0] + ".png")  # assume mask is PNG

        # destination paths
        dst_img = os.path.join(dst_images_dir, filename)
        dst_mask = os.path.join(dst_masks_dir, os.path.splitext(filename)[0] + ".png")

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)

        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)
        print(filename)
        return f"Copied {filename}"
    except Exception as e:
        return f"Error {filename}: {e}"


if __name__ == "__main__":
    # list of image files (filter jpg/jpeg/png)
    files = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # multiprocessing pool
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(copy_file, files)

    # print summary
    for r in results:
        print(r)
