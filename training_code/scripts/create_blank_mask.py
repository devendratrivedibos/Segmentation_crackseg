import os
import cv2
import numpy as np
# Define paths
image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_SPLIT\TRAIN\IMAGES"
mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_SPLIT\TRAIN\MASKS"

# List all image filenames
image_filenames = os.listdir(image_folder)

# Loop and check for each image
for image_name in image_filenames:
    image_path = os.path.join(image_folder, image_name)

    # Skip non-files or hidden files
    if not os.path.isfile(image_path) or image_name.startswith('.'):
        continue

    mask_path = os.path.join(mask_folder, image_name)
    if not os.path.exists(mask_path):
        print(f"Mask not found for image: {image_name}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_name}")
            continue

        height, width = image.shape[:2]

        # Create black mask (0, 0, 0)
        blank_mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Save the black mask
        cv2.imwrite(mask_path, blank_mask)