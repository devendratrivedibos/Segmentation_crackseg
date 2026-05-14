import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================
# CONFIG
# =====================
target_size = (4173, 10339)  # (width, height)

image_dir = r"Z:\Devendra\CONCRETE\HIGH_RES_IMAGES"
mask_dir = r"Z:\Devendra\CONCRETE\HIGH_RES_IMAGES"

out_image_dir = r"Z:\Devendra\CONCRETE\HIGH_RES_IMAGES"
out_mask_dir = r"Z:\Devendra\CONCRETE\HIGH_RES_IMAGES"

max_workers = 6   # adjust CPU usage

valid_img_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# create output folders
os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)


# =====================
# PROCESS FUNCTION
# =====================
def process_pair(img_file):

    if not img_file.lower().endswith(valid_img_ext):
        return None

    base_name = os.path.splitext(img_file)[0]

    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, base_name + ".png")

    if not os.path.exists(mask_path):
        return f"Mask missing: {base_name}.png"

    out_img_path = os.path.join(out_image_dir, img_file)
    out_mask_path = os.path.join(out_mask_dir, base_name + ".png")

    # read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return f"Image read error: {img_file}"

    # read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return f"Mask read error: {base_name}.png"

    # resize
    resized_img = cv2.resize(
        img,
        target_size,
        interpolation=cv2.INTER_LINEAR
    )

    resized_mask = cv2.resize(
        mask,
        target_size,
        interpolation=cv2.INTER_NEAREST
    )

    # save
    cv2.imwrite(out_img_path, resized_img)
    cv2.imwrite(out_mask_path, resized_mask)

    return f"Done: {base_name}"


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_img_ext)
    ]

    total_files = len(image_files)

    print(f"\nTotal images found: {total_files}\n")

    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(process_pair, img_file)
            for img_file in image_files
        ]
        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result:
                print(result)

            print(f"Progress: {completed}/{total_files}")


    print("\n✅ Resize completed for images + masks!")
#
#
#
#
import cv2
#
# img = cv2.imread(
#     r"Z:\Devendra\CONCRETE\HIGH_RES_IMAGES\3 - Copy.jpg",
#     cv2.IMREAD_COLOR
# )
#
# # width = 4173, height = 10339
# img = cv2.resize(
#     img,
#     (4173, 10339),
#     interpolation=cv2.INTER_LINEAR
# )
#
# cv2.imwrite(
#     r"Z:\Devendra\CONCRETE\HIGH_RES_IMAGES\IMG_20240917_122000_resized.jpg",
#     img
# )