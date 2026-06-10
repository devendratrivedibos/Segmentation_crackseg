import os
import shutil

# ====== CONFIG ======
images_dir = r"Z:\BHOPAL_MAY 2026\JAMUNIYA-BARELY_2026-05-15_11-44-18\SECTION-5\process_distress"
masks_dir = r"Z:\BHOPAL_MAY 2026\JAMUNIYA-BARELY_2026-05-15_11-44-18\SECTION-5\MASKS_NEW"

output_images = r"Z:\BHOPAL_MAY 2026\JAMUNIYA-BARELY_2026-05-15_11-44-18\SECTION-5\IMAGES_NEW"
output_masks = r"Z:\BHOPAL_MAY 2026\JAMUNIYA-BARELY_2026-05-15_11-44-18\SECTION-5\MASKS_NEW"
# Create output folders if they don't exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

# Get all image/mask names without extension
image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)}
mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir)}

# Find intersection
common = set(image_files.keys()) & set(mask_files.keys())

print(f"Found {len(common)} matches")

# Copy files
for name in common:
    img_src = os.path.join(images_dir, image_files[name])
    mask_src = os.path.join(masks_dir, mask_files[name])

    img_dst = os.path.join(output_images, image_files[name])
    mask_dst = os.path.join(output_masks, mask_files[name])

    shutil.copy2(img_src, img_dst)
    # shutil.copy2(mask_src, mask_dst)

    print(f"Copied: {image_files[name]} & {mask_files[name]}")


