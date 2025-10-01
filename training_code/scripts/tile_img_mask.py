import os
import cv2
import re

def tile_dataset(image_dir, mask_dir, save_dir_img, save_dir_mask,
                 tile_size=(1024, 1024), overlap=128):
    """
    Tile all images and masks in given folders into smaller crops with overlap.

    Args:
        image_dir (str): Folder with images.
        mask_dir (str): Folder with masks (same names as images).
        save_dir_img (str): Output folder for image tiles.
        save_dir_mask (str): Output folder for mask tiles.
        tile_size (tuple): (height, width) for tiles.
        overlap (int): Overlap in pixels between tiles.
    """
    os.makedirs(save_dir_img, exist_ok=True)
    os.makedirs(save_dir_mask, exist_ok=True)

    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    # img_files = []
    # for f in os.listdir(image_dir):
    #     if f.lower().endswith(('.jpg', '.jpeg', '.png')):
    #         match = re.search(r'-(\d+)-', f)  # extract number between dashes
    #         if match:
    #             num = int(match.group(1))
    #             if num > 344:
    #                 img_files.append(f)
    # img_files = sorted(img_files)
    print(len(img_files))
    th, tw = tile_size

    for fname in img_files:
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")

        if not os.path.exists(mask_path):
            print(f"⚠️ No mask found for {fname}, skipping.")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        H, W = img.shape[:2]
        step_y = th - overlap
        step_x = tw - overlap

        tile_id = 0
        base_name = os.path.splitext(fname)[0]

        for y in range(0, H, step_y):
            for x in range(0, W, step_x):
                y1, y2 = y, min(y + th, H)
                x1, x2 = x, min(x + tw, W)

                # Adjust to maintain fixed tile size at edges
                if y2 - y1 < th:
                    y1 = max(0, y2 - th)
                    y2 = y1 + th
                if x2 - x1 < tw:
                    x1 = max(0, x2 - tw)
                    x2 = x1 + tw

                img_tile = img[y1:y2, x1:x2]
                mask_tile = mask[y1:y2, x1:x2]

                img_filename = os.path.join(save_dir_img, f"{base_name}_{tile_id:05d}.jpg")
                mask_filename = os.path.join(save_dir_mask, f"{base_name}_{tile_id:05d}.png")

                cv2.imwrite(img_filename, img_tile)
                cv2.imwrite(mask_filename, mask_tile)

                tile_id += 1

        print(f"✅ {fname}: saved {tile_id} tiles.")



for i in range(1,2):
    path = rf"F:\NSV_DATA\Demo Data\SA DATA\CHAS-RAMGARH_2024-11-14_11-09-22\SECTION-{i}"
    if os.path.isdir(path) is False:
        print(f"❌ Path does not exist: {path}")
        continue
    image_dir = os.path.join(path, "process_distress_HIGH_RES")  # input image folder
    mask_dir = os.path.join(path, "HIGH_RES_MASKS") # input mask folder
    save_dir_img = os.path.join(path, "TILE_HIGH_RES_IMAGES")
    save_dir_mask = os.path.join(path, "TILE_HIGH_RES_MASKS")
    tile_dataset(image_dir, mask_dir, save_dir_img, save_dir_mask,
                 tile_size=(1024, 1024), overlap=256)
