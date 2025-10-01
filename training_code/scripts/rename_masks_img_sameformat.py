import os
import shutil
import re


folder = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\MASKS_4040"
camera_id = "404020"
#
# for fname in os.listdir(folder):
#     old_path = os.path.join(folder, fname)
#
#     if os.path.isfile(old_path) and fname.startswith("A_T_1_rangeDataFiltered"):
#         # match pattern: -0000020-
#         match = re.search(r"-(\d+)-", fname)
#         if match:
#             number = match.group(1)
#
#             # build new name
#             new_name = fname.replace(f"-{number}-", f"-_IMG_{camera_id}_{number}-")
#             new_path = os.path.join(folder, new_name)
#
#             print(f"{fname}  -->  {new_name}")
#             try:
#                 os.rename(old_path, new_path)
#             except Exception as e:
#                 continue






######################
import os
import re

folder = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\MASKS_4030"
camera_id = "403030"

for fname in os.listdir(folder):
    old_path = os.path.join(folder, fname)

    if os.path.isfile(old_path) and fname.startswith("AMRAVTI-TALEGAON"):
        # match ..._IMG_<number>
        match = re.match(r"(.+_IMG_)(\d+)(\..+)?", fname)
        if match:
            prefix, number, ext = match.groups()
            if ext is None:
                ext = ""  # handle missing extension
            # skip if aLready renamed
            if camera_id in fname:
                continue
            # new name with camera_id inserted
            new_name = f"{prefix}{camera_id}_{number}{ext}"
            new_path = os.path.join(folder, new_name)
            print(f"{fname}  -->  {new_name}")
            os.rename(old_path, new_path)
#########################3




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


import os
import re

# --- CONFIG ---
folder = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_highres"   # <<< change this
for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)

    if not os.path.isfile(old_path):
        continue

    # only fix files without an extension
    if "." not in filename:
        new_name = filename + ".jpg"
        new_path = os.path.join(folder, new_name)

        print(f"Fixing: {filename}  -->  {new_name}")
        os.rename(old_path, new_path)
