import os
import shutil

# --- Directories ---
img_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\IMAGES_4030"
mask_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4030"

# --- Collect names without extension ---
img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir)}

# --- Find mismatched names ---
only_in_img = img_files - mask_files
only_in_mask = mask_files - img_files
mismatched = only_in_img.union(only_in_mask)
print(len(only_in_mask), len(only_in_img))

print(f"Found {len(mismatched)} matched files", mismatched)
# --- Copy matching originals from OLD dataset ---

print(f"Found {len(mismatched)} mismatched files")

# --- Delete extra files ---
for f in os.listdir(img_dir):
    name, ext = os.path.splitext(f)
    if name in only_in_img:
        os.remove(os.path.join(img_dir, f))
        print(f"Deleted from IMAGES: {f}")

for f in os.listdir(mask_dir):
    name, ext = os.path.splitext(f)
    if name in only_in_mask:
        os.remove(os.path.join(mask_dir, f))
        print(f"Deleted from MASKS: {f}")