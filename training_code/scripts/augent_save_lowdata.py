import cv2
import os
import random
# Input folders
images_folder =  r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\AnnotationMasks_COPY"
masks_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\AnnotationMasks_COPY"

# Output folders (must be different from input to avoid overwrite)
aug_images_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\DATA\AnnotationImages"
aug_masks_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\DATA\AnnotationMasks"

os.makedirs(aug_images_folder, exist_ok=True)
os.makedirs(aug_masks_folder, exist_ok=True)

image_files = sorted(os.listdir(images_folder))
mask_files = sorted(os.listdir(masks_folder))

for img_file, mask_file in zip(image_files, mask_files):
    flip_choices = [None, 1, 0, -1]
    flip_code = random.choice(flip_choices)
    if flip_code is not None:
        print(f"Flipping {img_file} and {mask_file} with code {flip_code}")
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
