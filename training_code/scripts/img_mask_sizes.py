import os
import shutil
img_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\ACCEPTED_IMAGES"
mask_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\ACCEPTED_MASKS"

# Get files without extension
img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir)}
print(len(img_files))
print(len(mask_files))
# Differences
only_in_img = img_files - mask_files
only_in_mask = mask_files - img_files
mismatched = only_in_img.union(only_in_mask)
common = img_files & mask_files

print("Only in Images:", only_in_img)
print("Only in Masks:", only_in_mask)
