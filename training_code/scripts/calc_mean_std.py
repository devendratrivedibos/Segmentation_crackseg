import cv2
import os
import numpy as np
from tqdm import tqdm

image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_IMAGES"

# List all image files
image_files = [os.path.join(image_folder, f)
               for f in os.listdir(image_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Accumulators
mean = np.zeros(3)
std = np.zeros(3)
total_pixels = 0

for img_path in tqdm(image_files):
    img = cv2.imread(img_path)          # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # normalize to 0-1
    h, w, c = img.shape
    total_pixels += h * w

    mean += img.sum(axis=(0, 1))
    std += ((img - img.mean(axis=(0,1)))**2).sum(axis=(0,1))

mean /= total_pixels
std = np.sqrt(std / total_pixels)

print("Mean:", mean)
print("Std:", std)
