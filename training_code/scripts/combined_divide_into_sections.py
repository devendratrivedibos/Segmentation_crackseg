import os
import shutil

# ================== CONFIG ==================
images_dir = r"T:\SHINGOTE-KOLHAR_2025-09-23_14-06-00\AnnotationImages"   # folder containing images
masks_dir = r"T:\SHINGOTE-KOLHAR_2025-09-23_14-06-00\AnnotationMasks"     # folder containing masks
section = "SECTION-9"  # section to copyy
output_dir = fr"T:\SHINGOTE-KOLHAR_2025-09-23_14-06-00\{section}"  # destination folder

# create destination subfolders
os.makedirs(os.path.join(output_dir, "AnnotationImages"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "AnnotationMasks"), exist_ok=True)

# ================== COPY SECTION-1 FILES ==================
for filename in os.listdir(images_dir):
    if section in filename:
        img_path = os.path.join(images_dir, filename)
        pngmask = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(masks_dir, pngmask)

        if os.path.exists(mask_path):  # ensure both exist
            shutil.copy2(img_path, os.path.join(output_dir, "AnnotationImages", filename))
            shutil.copy2(mask_path, os.path.join(output_dir, "AnnotationMasks", filename))
            print(f"Copy: {filename}")
        else:
            # print(f"Mask not found for {filename}")
            pass
print("✅ All matching files moved.")
