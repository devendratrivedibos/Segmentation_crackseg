import cv2
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# =================== CONFIG ===================
images_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\OG_DATASET_CONCRETE\ACCEPTED_IMAGES_COPY"
masks_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\OG_DATASET_CONCRETE\ACCEPTED_MASKS_COPY"

aug_images_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\OG_DATASET_CONCRETE\ACCEPTED_IMAGES_COPY"
aug_masks_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\OG_DATASET_CONCRETE\ACCEPTED_MASKS_COPY"

os.makedirs(aug_images_folder, exist_ok=True)
os.makedirs(aug_masks_folder, exist_ok=True)

# =================== WORKER FUNCTION ===================
def process_pair(img_file, mask_file):
    flip_choices = [None, 1, 0, -1]
    flip_code = random.choice(flip_choices)
    if flip_code is None:
        return f"Skipped {img_file}"

    img_path = os.path.join(images_folder, img_file)
    mask_path = os.path.join(masks_folder, mask_file)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if image is None or mask is None:
        return f"Error loading {img_file} or {mask_file}"

    flipped_image = cv2.flip(image, flip_code)
    flipped_mask = cv2.flip(mask, flip_code)

    cv2.imwrite(os.path.join(aug_images_folder, img_file), flipped_image)
    cv2.imwrite(os.path.join(aug_masks_folder, mask_file), flipped_mask)

    return f"Flipped {img_file} with code {flip_code}"

# =================== MAIN PROCESS ===================
if __name__ == "__main__":
    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(masks_folder))

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_pair, img_file, mask_file)
            for img_file, mask_file in zip(image_files, mask_files)
        ]

        for future in as_completed(futures):
            print(future.result())
