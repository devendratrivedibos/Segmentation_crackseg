import os
import shutil

# ================== CONFIG ==================
images_dir = r"W:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\only_JS_IMAGE"   # folder containing images
masks_dir = r"W:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\only_JS_MASK"     # folder containing masks
output_dir = r"W:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-1"  # destination folder

# create destination subfolders
os.makedirs(os.path.join(output_dir, "REWORK_IMAGES"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "REWORK_MASKS"), exist_ok=True)

# ================== MOVE SECTION-1 FILES ==================
for filename in os.listdir(images_dir):
    if "SECTION-1" in filename:
        img_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, filename)

        if os.path.exists(mask_path):  # ensure both exist
            shutil.move(img_path, os.path.join(output_dir, "images", filename))
            shutil.move(mask_path, os.path.join(output_dir, "masks", filename))
            print(f"Moved: {filename}")
        else:
            print(f"Mask not found for {filename}")

print("✅ All matching files moved.")
