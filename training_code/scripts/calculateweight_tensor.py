import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------
# CONFIG
# --------------------------
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
    (112, 102, 255): 12,  # Popout
    (255, 255, 255): 13,  # Unclassified
    (255, 215, 0): 14,    # Cracking
}

NUM_CLASSES = len(set(COLOR_MAP.values()))
mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\DATA\DATASET_SPLIT\TRAIN\MASKS"

# --------------------------
# Worker function
# --------------------------
def process_mask(file_path):
    """Process one mask → return class pixel counts"""
    mask = cv2.imread(file_path)
    if mask is None:
        return np.zeros(NUM_CLASSES, dtype=np.int64)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Initialize counts for this mask
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    # Map each color to class ID
    mask_ids = np.zeros(mask.shape[:2], dtype=np.int64)
    for color, cls_id in COLOR_MAP.items():
        matches = np.all(mask == color, axis=-1)
        mask_ids[matches] = cls_id

    # Count pixels for each class
    for cls_id in range(NUM_CLASSES):
        counts[cls_id] = np.sum(mask_ids == cls_id)

    return counts


# --------------------------
# Main threaded loop
# --------------------------
def count_pixels_parallel(folder, num_workers=8):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")]
    total_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_mask, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing masks"):
            counts = future.result()
            total_counts += counts

    return total_counts


# --------------------------
# Run counting
# --------------------------
if __name__ == "__main__":
    print("🔍 Counting pixels per class (threaded)...")
    class_counts = count_pixels_parallel(mask_folder, num_workers=8)

    # Compute inverse frequency weights
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (NUM_CLASSES * class_counts)

    # Convert to tensor for PyTorch
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print("\n📊 Class counts:", class_counts)
    print("⚖ Class weights:", class_weights_tensor)
