import os
import shutil
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===================== CONFIG =====================
images_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\TILE_HIGH_RES_IMAGES"
masks_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\TILE_HIGH_RES_MASKS"
output_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\SPLITTED"

test_size = 0.10
val_size = 0.15
random_state = 42
max_workers = 8   # adjust based on your CPU cores

# --- Color map (RGB) → ID ---
COLOR_MAP = {
    (0, 0, 0): 0,         # Background
    (255, 0, 0): 1,       # Alligator
    (0, 0, 255): 2,       # Transverse Crack
    (0, 255, 0): 3,       # Longitudinal Crack
    (139, 69, 19): 4,     # Pothole
    (255, 165, 0): 5,     # Patches
    (255, 0, 255): 6,     # Multiple Crack
    (0, 255, 255): 7,     # Spalling
    (0, 128, 0): 8,       # Corner Break
    (255, 100, 203): 9,   # Sealed Joint - T
    (199, 21, 133): 10,   # Sealed Joint - L
    (128, 0, 128): 11,    # Punchout
    (112, 102, 255): 12,  # Popout Grey
    (255, 255, 255): 13,  # Unclassified
    (255, 215, 0): 14,    # Cracking
}

NUM_CLASSES = len(COLOR_MAP)

# ===================== LUT for fast RGB → class_id =====================
lut = np.full((256, 256, 256), -1, dtype=np.int16)
for rgb, cls_id in COLOR_MAP.items():
    lut[rgb] = cls_id


# ===================== UTILITIES =====================
def get_label_vector(mask_path):
    """Convert mask → multilabel vector"""
    mask = np.array(Image.open(mask_path).convert("RGB"))
    mask_cls = lut[mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]]
    unique_cls = np.unique(mask_cls)
    label_vec = np.zeros(NUM_CLASSES, dtype=int)
    for cls in unique_cls:
        if cls >= 0:
            label_vec[cls] = 1
    return label_vec


def process_mask_file(f):
    """Process one mask file: return (filename, label_vector)"""
    if not f.endswith(".png"):
        return None
    mask_path = os.path.join(masks_dir, f)
    label_vec = get_label_vector(mask_path)

    base = os.path.splitext(f)[0]
    if os.path.exists(os.path.join(images_dir, base + ".jpg")):
        img_name = base + ".jpg"
    elif os.path.exists(os.path.join(images_dir, base + ".png")):
        img_name = base + ".png"
    else:
        return None
    return img_name, label_vec


def copy_pair(f, split):
    """Copy image & mask pair"""
    base = os.path.splitext(f)[0]
    img_src = os.path.join(images_dir, f)
    mask_src = os.path.join(masks_dir, base + ".png")

    img_out = os.path.join(output_dir, split, "IMAGES", f)
    mask_out = os.path.join(output_dir, split, "MASKS", base + ".png")

    if os.path.exists(img_src) and os.path.exists(mask_src):
        shutil.copy2(img_src, img_out)
        shutil.copy2(mask_src, mask_out)


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


# ===================== MAIN =====================
if __name__ == "__main__":

    # -------- STEP 1: Load all masks & labels (Multiprocessing) --------
    print("Extracting labels from masks (parallel)...")
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

    filenames, labels = [], []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_mask_file, f) for f in mask_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Masks"):
            result = future.result()
            if result is not None:
                img_name, label_vec = result
                filenames.append(img_name)
                labels.append(label_vec)

    if len(filenames) == 0:
        raise ValueError("No matching images and masks found. Check your directories!")
    labels = np.array(labels)

    # -------- STEP 2: Stratified Split --------
    print("Performing stratified split...")
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(msss1.split(filenames, labels))

    val_ratio = val_size / (1 - test_size)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(msss2.split([filenames[i] for i in train_val_idx],
                                          labels[train_val_idx]))

    train_files = [filenames[i] for i in train_val_idx[train_idx]]
    val_files = [filenames[i] for i in train_val_idx[val_idx]]
    test_files = [filenames[i] for i in test_idx]

    splits = {"TRAIN": train_files, "VAL": val_files, "TEST": test_files}

    # -------- STEP 3: Copy files (Multiprocessing) --------
    print("Copying files into dataset/ ...")
    for split, files in splits.items():
        img_out_dir = os.path.join(output_dir, split, "IMAGES")
        mask_out_dir = os.path.join(output_dir, split, "MASKS")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(lambda f: copy_pair(f, split), files),
                total=len(files),
                desc=f"Copy {split}"
            ))

    # -------- STEP 4: Count distribution --------
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
