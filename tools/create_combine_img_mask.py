import os
import shutil

# Define paths
base_path = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined"
output_images = os.path.join(base_path, "DATASET_IMAGES")
output_masks = os.path.join(base_path, "DATASET_MASKS")

# Ensure output dirs exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

# Contributors and their folders
contributors = ["Ankit", "Devendra",
                "Gaurav", "Jayesh", "Payal",
                "Sagar", "sam", "Sanket",
                "Saurav", "Shiva",
                "Vishwas"]

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

