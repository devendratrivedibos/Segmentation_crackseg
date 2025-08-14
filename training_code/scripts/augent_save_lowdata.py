import cv2
import os

# Input folders
images_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_IMAGES_COPY"
masks_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_MASKS_COPY"

# Output folders (must be different from input to avoid overwrite)
aug_images_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_IMAGES_COPY"
aug_masks_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_MASKS_COPY"

os.makedirs(aug_images_folder, exist_ok=True)
os.makedirs(aug_masks_folder, exist_ok=True)

image_files = sorted(os.listdir(images_folder))
mask_files = sorted(os.listdir(masks_folder))

for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(images_folder, img_file)
    mask_path = os.path.join(masks_folder, mask_file)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Flip horizontally
    flipped_image = cv2.flip(image, -1)
    flipped_mask = cv2.flip(mask, -1)

    # Save with the same filenames in augmented folders
    cv2.imwrite(os.path.join(aug_images_folder, img_file), flipped_image)
    cv2.imwrite(os.path.join(aug_masks_folder, mask_file), flipped_mask)
