import os
import shutil
import random

# Paths
images_dir = r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_IMAGES'
masks_dir = r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_MASKS'
output_base = r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_SPLIT'
split_ratio = 0.8  # 80% train, 20% val

# Get image filenames
image_filenames = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle and split
random.shuffle(image_filenames)
split_index = int(len(image_filenames) * split_ratio)
train_files = image_filenames[:split_index]
val_files = image_filenames[split_index:]

# Create folders
for split in ['TRAIN', 'VAL']:
    os.makedirs(os.path.join(output_base, split, 'IMAGES'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'MASKS'), exist_ok=True)


# Move files
def move_pairs(file_list, split):
    for filename in file_list:
        img_src = os.path.join(images_dir, filename)
        mask_src = os.path.join(masks_dir, filename)
        img_dst = os.path.join(output_base, split, 'IMAGES', filename)
        mask_dst = os.path.join(output_base, split, 'MASKS', filename)

        if os.path.exists(img_src) and os.path.exists(mask_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)
        else:
            print(f"Skipping {filename} (image or mask missing)")


move_pairs(train_files, 'TRAIN')
move_pairs(val_files, 'VAL')

print(f"Split complete: {len(train_files)} train, {len(val_files)} val")
