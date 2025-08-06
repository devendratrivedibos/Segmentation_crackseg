import os
import shutil

# Define paths
# base_path = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined"
# output_images = os.path.join(base_path, "DATASET_IMAGES")
# output_masks = os.path.join(base_path, "DATASET_MASKS")
#
# # Ensure output dirs exist
# os.makedirs(output_images, exist_ok=True)
# os.makedirs(output_masks, exist_ok=True)
#
# # Contributors and their folders
# contributors = ["Ankit", "Devendra",
#                 "Gaurav", "Jayesh", "Payal",
#                 "Sagar", "sam", "Sanket",
#                 "Saurav", "Shiva",
#                 "Vishwas"]
#
# for name in contributors:
#     img_dir = os.path.join(base_path, name)
#     mask_dir = os.path.join(img_dir, "Masks")
#
#     for fname in os.listdir(img_dir):
#         img_path = os.path.join(img_dir, fname)
#         mask_path = os.path.join(mask_dir, fname)
#
#         # Check if it's a file and a corresponding mask exists
#         if os.path.isfile(img_path) and os.path.isfile(mask_path):
#             # Copy to final dataset folders
#             shutil.copy2(img_path, os.path.join(output_images, fname))
#             shutil.copy2(mask_path, os.path.join(output_masks, fname))
#
# print("✅ Images and corresponding masks copied to DATASET_IMAGES and DATASET_MASKS.")
# print(len(os.listdir(output_images)))
# print(len(os.listdir(output_masks)))



import os
import shutil

# Paths
combined_folder = r"D:\cracks\3Channel\Semantic-Segmentation of pavement distress dataset\Combined"
masks_folder = r"D:\cracks\3Channel\Semantic-Segmentation of pavement distress dataset\MASKS"

images_folder = r"D:\cracks\3Channel\Semantic-Segmentation of pavement distress dataset\IMAGES"

# Ensure destination folder exists
os.makedirs(images_folder, exist_ok=True)

# Loop through files in combined
# Loop through .jpg files in combined
for file_name in os.listdir(combined_folder):
    if file_name.lower().endswith(".jpg"):
        base_name = os.path.splitext(file_name)[0]  # Remove .jpg
        mask_file = base_name + ".png"
        mask_path = os.path.join(masks_folder, mask_file)

        if os.path.exists(mask_path):
            src_path = os.path.join(combined_folder, file_name)
            dst_path = os.path.join(images_folder, file_name)
            shutil.copy2(src_path, dst_path)
            # print(f"Copied: {file_name}")
        else:
            print(f"No matching mask for: {file_name}")

            import os

            # Paths

# Get base names without extensions
image_basenames = {
    os.path.splitext(f)[0]
    for f in os.listdir(images_folder)
    if f.lower().endswith(".jpg")
}

mask_basenames = {
    os.path.splitext(f)[0]
    for f in os.listdir(masks_folder)
    if f.lower().endswith(".png")
}

# Find mismatches
images_not_in_masks = image_basenames - mask_basenames
masks_not_in_images = mask_basenames - image_basenames

image_basenames = {
    os.path.splitext(f)[0]
    for f in os.listdir(images_folder)
    if f.lower().endswith(".jpg")
}

# Loop through mask files and delete if no corresponding image
deleted = []
for f in os.listdir(masks_folder):
    if f.lower().endswith(".png"):
        base = os.path.splitext(f)[0]
        if base not in image_basenames:
            path_to_delete = os.path.join(masks_folder, f)
            os.remove(path_to_delete)
            deleted.append(f)

# Report deleted files
print("Deleted mask files:")
for f in deleted:
    print(f)
