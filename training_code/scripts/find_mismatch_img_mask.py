import os
import shutil

# --- Directories ---
img_dir = r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-3\ACCEPTED_MASKS"
mask_dir = r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-3\ACCEPTED_IMAGES"

# --- Collect names without extension ---
img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir)}

# --- Find mismatched names ---
only_in_img = img_files - mask_files
only_in_mask = mask_files - img_files
mismatched = only_in_img.union(only_in_mask)

print(f"Found {len(mismatched)} mismatched files", mismatched)

# --- Copy matching originals from OLD dataset ---
