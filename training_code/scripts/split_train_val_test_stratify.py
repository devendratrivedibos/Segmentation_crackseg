import os
import shutil
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm

# ===================== CONFIG =====================
images_dir = r"V:\Devendra\ASPHALT_ACCEPTED\COMBINED_IMAGES"
masks_dir = r"V:\Devendra\ASPHALT_ACCEPTED\COMBINED_MASKS"
output_dir = r"V:\Devendra\ASPHALT_ACCEPTED\COMBINED_SPLITTED"


test_size = 0.01
val_size = 0.20
random_state = 42

# --- Color map (RGB) → ID ---
COLOR_MAP = {
    (0, 0, 0): 0,         # Black - Background
    (255, 0, 0): 1,       # Red - Alligator
    (0, 0, 255): 2,       # Blue - Transverse Crack
    (0, 255, 0): 3,       # Green - Longitudinal Crack
    (139, 69, 19): 4,     # Brown - Pothole
    (255, 165, 0): 5,     # Orange - Patches
    (255, 0, 255): 6,    # Violet - Multiple Crack
    (0, 255, 255): 7,     # Cyan - Spalling
    (0, 128, 0): 8,       # Dark Green - Corner Break
    (255, 100, 203): 9,   # Light Pink - Sealed Joint - T
    (199, 21, 133): 10,   # Dark Pink - Sealed Joint - L
    (128, 0, 128): 11,     # Purple - Punchout
    (112, 102, 255): 12,  #popout Grey
    (255, 255, 255): 13,  # White - Unclassified
    (255, 215, 0): 14,  # Gold - Cracking
}


NUM_CLASSES = len(COLOR_MAP)

# ===================== LUT for fast RGB → class_id =====================
lut = np.full((256, 256, 256), -1, dtype=np.int16)
for rgb, cls_id in COLOR_MAP.items():
    lut[rgb] = cls_id


def get_label_vector(mask_path):
    """Convert mask → multilabel vector"""
    mask = Image.open(mask_path).convert("RGB")
    # mask = mask.resize((419, 1024), Image.NEAREST)  # ✅ works
    mask = np.array(mask)  # convert to NumPy after resizing
    mask_cls = lut[mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]]
    unique_cls = np.unique(mask_cls)
    label_vec = np.zeros(NUM_CLASSES, dtype=int)
    for cls in unique_cls:
        if cls >= 0:
            label_vec[cls] = 1
    return label_vec


def count_classes_in_dir(mask_dir):
    """Count images containing each class"""
    counts = Counter()
    for f in os.listdir(mask_dir):
        if not f.endswith(".png"):
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, f)).convert("RGB"))
        mask_cls = lut[mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]]
        unique_cls = np.unique(mask_cls)
        for cls in unique_cls:
            if cls >= 0:
                counts[cls] += 1
    return counts


# ===================== STEP 1: Load all masks & labels =====================
filenames, labels = [], []
print("Extracting labels from masks...")
for f in tqdm(os.listdir(masks_dir)):
    if not f.endswith(".png"):
        continue
    mask_path = os.path.join(masks_dir, f)
    label_vec = get_label_vector(mask_path)

    base = os.path.splitext(f)[0]  # filename without extension
    # check if image exists as jpg or png
    if os.path.exists(os.path.join(images_dir, base + ".jpg")):
        img_name = base + ".jpg"
    elif os.path.exists(os.path.join(images_dir, base + ".png")):
        img_name = base + ".png"
    else:
        continue  # skip if no matching image

    filenames.append(img_name)
    labels.append(label_vec)
if len(filenames) == 0:
    raise ValueError("No matching images and masks found. Check your directories!")
labels = np.array(labels)

# ===================== STEP 2: Stratified Split =====================
print("Performing stratified split...")
msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
train_val_idx, test_idx = next(msss1.split(filenames, labels))

# split train/val
val_ratio = val_size / (1 - test_size)  # relative val size
msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
train_idx, val_idx = next(msss2.split([filenames[i] for i in train_val_idx],
                                      labels[train_val_idx]))

train_files = [filenames[i] for i in train_val_idx[train_idx]]
val_files = [filenames[i] for i in train_val_idx[val_idx]]
test_files = [filenames[i] for i in test_idx]

splits = {"TRAIN": train_files, "VAL": val_files, "TEST": test_files}

# ===================== STEP 3: Move files =====================
print("Copying files into dataset/ ...")
for split, files in splits.items():
    img_out = os.path.join(output_dir, split, "IMAGES")
    mask_out = os.path.join(output_dir, split, "MASKS")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    for f in tqdm(files, desc=split):
        base = os.path.splitext(f)[0]
        img_src = os.path.join(images_dir, f)
        mask_src = os.path.join(masks_dir, base + ".png")

        if os.path.exists(img_src) and os.path.exists(mask_src):
            shutil.copy2(img_src, os.path.join(img_out, f))
            shutil.copy2(mask_src, os.path.join(mask_out, base + ".png"))

# ===================== STEP 4: Count distribution =====================
print("\nClass Distribution (per split):")
split_counts = defaultdict(Counter)
for split in splits.keys():
    mask_dir = os.path.join(output_dir, split, "MASKS")
    split_counts[split] = count_classes_in_dir(mask_dir)

for split in splits.keys():
    print(f"\n{split}:")
    total_images = len(os.listdir(os.path.join(output_dir, split, "MASKS")))
    print(f"  Total images: {total_images}")
    for cls_id in range(NUM_CLASSES):
        count = split_counts[split].get(cls_id, 0)
        pct = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  Class {cls_id}: {count} images ({pct:.2f}%)")
