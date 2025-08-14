import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Color map for your crack segmentation
COLOR_MAP = {
    (0, 0, 0): 0,      # Background
    (255, 0, 0): 1,    # Alligator Crack
    (0, 0, 255): 2,    # Transverse Crack
    (0, 255, 0): 3,    # Longitudinal Crack
    (255, 0, 255): 4,  # Multiple Crack
    (255, 204, 0): 5,  # Joint Seal
    (0, 42, 255): 6    # Pothole
}

NUM_CLASSES = len(set(COLOR_MAP.values()))
mask_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_MASKS"

# Pixel count for each class
class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

print("🔍 Counting pixels per class...")
for file in tqdm(os.listdir(mask_folder)):
    if file.lower().endswith(".png"):
        mask = cv2.imread(os.path.join(mask_folder, file))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Map each color to class ID
        mask_ids = np.zeros(mask.shape[:2], dtype=np.int64)
        for color, cls_id in COLOR_MAP.items():
            matches = np.all(mask == color, axis=-1)
            mask_ids[matches] = cls_id

        # Count pixels
        for cls_id in range(NUM_CLASSES):
            class_counts[cls_id] += np.sum(mask_ids == cls_id)

# Compute inverse frequency weights
total_pixels = class_counts.sum()
class_weights = total_pixels / (NUM_CLASSES * class_counts)

# Convert to tensor for PyTorch
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print("\n📊 Class counts:", class_counts)
print("⚖ Class weights:", class_weights_tensor)
