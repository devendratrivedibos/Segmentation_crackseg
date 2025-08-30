import os
import shutil

# Define paths
"""
base_path = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined"

output_images = os.path.join(base_path, "DATASET_IMAGES_")
output_masks = os.path.join(base_path, "DATASET_MASKS_")

# Ensure output dirs exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

# Contributors and their folders
contributors = ["segmentation_dataset_08_aug"]

for name in contributors:
    img_dir = os.path.join(base_path, name)
    mask_dir = os.path.join(img_dir, "Masks")

    for fname in os.listdir(img_dir):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        # Check if it's a file and a corresponding mask exists
        if os.path.isfile(img_path) and os.path.isfile(mask_path):
            # Copy to final dataset folders
            shutil.copy2(img_path, os.path.join(output_images, fname))
            shutil.copy2(mask_path, os.path.join(output_masks, fname))

print("✅ Images and corresponding masks copied to DATASET_IMAGES and DATASET_MASKS.")
print(len(os.listdir(output_images)))
print(len(os.listdir(output_masks)))
"""

import os
import shutil

# Paths
base_path = r"D:\cracks\Semantic-Segmentation of pavement distress dataset"
base_path = r"X:\DataSet"
output_images = os.path.join(base_path, "DATASET_IMAGES_")
output_masks = os.path.join(base_path, "DATASET_MASKS_")

# Ensure output dirs exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

# Contributors (subfolders)
contributors = ["Combined_Images"]
# contributors = ["Combined_OG_OLD"]
for name in contributors:
    img_dir = os.path.join(base_path, name)
    mask_dir = os.path.join(img_dir, "Masks")

    # Build a set of mask basenames for quick lookup
    mask_basenames = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) if f.lower().endswith(".png")}

    for fname in os.listdir(img_dir):
        if fname.lower().endswith(".jpg") or fname.lower().endswith(".png"):
            base_name = os.path.splitext(fname)[0]
            if base_name in mask_basenames:
                img_path = os.path.join(img_dir, fname)
                mask_path = os.path.join(mask_dir, mask_basenames[base_name])

                # Copy to output
                shutil.copy2(img_path, os.path.join(output_images, fname))
                shutil.copy2(mask_path, os.path.join(output_masks, mask_basenames[base_name]))
            else:
                print(f"❌ No matching mask for image: {fname}")

print("✅ Matching images and masks copied.")
print(f"Total Images: {len(os.listdir(output_images))}")
print(f"Total Masks: {len(os.listdir(output_masks))}")
