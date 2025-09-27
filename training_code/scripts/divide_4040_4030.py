import os
import shutil
from pathlib import Path

# Base path containing all SECTION folders
base_folder = r"z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51"

for section_name in os.listdir(base_folder):
    section_path = os.path.join(base_folder, section_name)
    if not os.path.isdir(section_path) or not section_name.lower().startswith("section-4"):
        continue
    print(f"Handling section: {section_name}")

    process_distress = os.path.join(section_path, "process_distress_40")
    process_4030 = os.path.join(section_path, "IMAGES_4030")
    process_4040 = os.path.join(section_path, "IMAGES_4040")

    os.makedirs(process_4030, exist_ok=True)
    os.makedirs(process_4040, exist_ok=True)

    if not os.path.exists(process_distress):
        continue  # skip if process_distress doesn't exist

    for f in os.listdir(process_distress):
        src = os.path.join(process_distress, f)
        if not os.path.isfile(src):
            continue

        if "IMG_4030" in f:
            dst = os.path.join(process_4030, f)
            shutil.move(src, dst)
        elif "IMG_4040" in f:
            dst = os.path.join(process_4040, f)
            shutil.move(src, dst)

    print(f"Processed {section_name}")


# --- CONFIG ---
# images_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\IMAGES_4030"
# masks_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4030"
# output_images = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\IMAGES_4030"
# output_masks = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\MASKS_4030"
# filter_key = "A_T_1"
#
# # Create output dirs
# os.makedirs(output_images, exist_ok=True)
# os.makedirs(output_masks, exist_ok=True)
#
# # Iterate through images
# for img_file in os.listdir(images_dir):
#     img_path = Path(images_dir) / img_file
#     name, ext = os.path.splitext(img_file)
#
#     if filter_key in img_file:
#         # expected mask
#         mask_path = Path(masks_dir) / f"{name}.png"
#
#         if mask_path.exists():
#             # copy image
#             shutil.copy2(img_path, Path(output_images) / img_file)
#             # copy mask
#             shutil.copy2(mask_path, Path(output_masks) / mask_path.name)
#         else:
#             print(f"⚠ Mask not found for: {img_file}")
# """
