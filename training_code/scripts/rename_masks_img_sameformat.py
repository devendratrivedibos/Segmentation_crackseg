import os
import shutil
import re

# ==== CONFIG ====
src_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4040"
dst_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4040"

os.makedirs(dst_folder, exist_ok=True)

for i in os.listdir(src_folder):
    old_path = os.path.join(src_folder, i)

    if os.path.isfile(old_path) and i.startswith("A_T_1_rangeDataFiltered"):
        # just replace A_T_1 with A_T_2
        new_name = i.replace("A_T_1_rangeDataFiltered-_IMG_404020_", "A_T_1_rangeDataFiltered-", 1)
        new_path = os.path.join(dst_folder, new_name)

        os.rename(old_path, new_path)


import os
# import re
#
# # update this to your folder containing all images
# img_folder = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\ALL_IMAGES"
# img_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4040"
# camera_id = "404020"
#
# for fname in os.listdir(img_folder):
#     old_path = os.path.join(img_folder, fname)
#
#     if os.path.isfile(old_path) and fname.startswith("AMRAVTI-TALEGAON"):
#         # match pattern: ..._IMG_<number>...
#         match = re.match(r"(.+_IMG_)(\d+)(\..+)?", fname)
#         if match:
#             prefix, number, ext = match.groups()
#             if ext is None:
#                 ext = ""  # in case no extension
#
#             # skip if camera_id already present
#             if camera_id in fname:
#                 continue
#
#             # new name with camera_id injected
#             new_name = f"{prefix}{camera_id}_{number}{ext}"
#             new_path = os.path.join(img_folder, new_name)
#
#             print(f"{fname}  -->  {new_name}")
#             os.rename(old_path, new_path)
