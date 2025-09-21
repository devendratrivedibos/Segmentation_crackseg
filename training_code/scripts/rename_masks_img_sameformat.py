import os
import shutil
import re

# ==== CONFIG ====
src_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\OG_DATASET_ASPHALT_OLD\AnnotationMasks"
dst_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4030"

os.makedirs(dst_folder, exist_ok=True)

# for f in os.listdir(src_folder):
#     old_path = os.path.join(src_folder, f)
#
#     if os.path.isfile(old_path) and f.startswith("A_T_3_rangeDataFiltered"):
#         # extract the number (e.g. 0000207)
#         match = re.search(r"-(\d+)-", f)
#         if match:
#             number = match.group(1)
#
#             # build new filename
#             new_name = f"A_T_3_rangeDataFiltered-_IMG_404020_{number}-_crack.png"
#             new_path = os.path.join(dst_folder, new_name)
#
#             # copy with new name
#             shutil.copy(old_path, new_path)
#             print(f"Copied & renamed: {f} → {new_name}")
#

# print(len(region_indexes))
# print(sorted(region_indexes))
# folder = r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-2\process_distress"
# for f in os.listdir(folder):
#     old_path = os.path.join(folder, f)
#
#     if os.path.isfile(old_path) and "A_T__" in f:
#         new_name = f.replace("A_T__", "A_T_2_")
#         new_path = os.path.join(folder, new_name)
#
#         os.rename(old_path, new_path)
#         print(f"Renamed: {f} → {new_name}")

# ==== CONFIG ====
src_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4030"
dst_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\Masks_4040"

os.makedirs(dst_folder, exist_ok=True)

for f in os.listdir(src_folder):
    old_path = os.path.join(src_folder, f)

    if os.path.isfile(old_path) and f.lower().endswith(".png"):
        # Replace A_T_<number> with A_T_4
        # new_name = re.sub(r"A_T_\d+", "A_T", f)

        # Replace IMG_403030 with IMG_404020 (if exists)
        new_name = f.replace("IMG_403030", "IMG_404020")

        new_path = os.path.join(dst_folder, new_name)

        shutil.copy(old_path, new_path)
        print(f"Copied & renamed: {f} → {new_name}")