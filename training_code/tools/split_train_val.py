import os
import shutil
import random

# Paths
images_dir = r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_IMAGES'
masks_dir = r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_MASKS_CLEANED'
output_base = r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_SPLIT'
split_ratio = 0.85  # 80% train, 20% val

# Get matching base filenames (only those that have both .jpg and .png)
image_filenames = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith('.png') and os.path.exists(os.path.join(masks_dir, os.path.splitext(f)[0] + '.png'))
]

# Shuffle and split
random.shuffle(image_filenames)
split_index = int(len(image_filenames) * split_ratio)
train_files = image_filenames[:split_index]
val_files = image_filenames[split_index:]

# Create folders
for split in ['TRAIN', 'VAL']:
    os.makedirs(os.path.join(output_base, split, 'IMAGES'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'MASKS'), exist_ok=True)

# Copy pairs
def copy_pairs(file_list, split):
    for img_filename in file_list:
        base_name = os.path.splitext(img_filename)[0]
        mask_filename = base_name + '.png'

        img_src = os.path.join(images_dir, img_filename)
        mask_src = os.path.join(masks_dir, mask_filename)

        img_dst = os.path.join(output_base, split, 'IMAGES', img_filename)
        mask_dst = os.path.join(output_base, split, 'MASKS', mask_filename)

        if os.path.exists(img_src) and os.path.exists(mask_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)
        else:
            print(f"Skipping: {img_filename} (missing image or mask)")

# Perform split copy
copy_pairs(train_files, 'TRAIN')
copy_pairs(val_files, 'VAL')

print(f"Split complete: {len(train_files)} train, {len(val_files)} val")

