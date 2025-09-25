import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\IMAGES_4040"

# List all image files
image_files = [os.path.join(image_folder, f)
               for f in os.listdir(image_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def process_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
        h, w, _ = img.shape
        pixel_count = h * w
        mean = img.sum(axis=(0, 1))
        std = ((img - img.mean(axis=(0, 1))) ** 2).sum(axis=(0, 1))
        return mean, std, pixel_count
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return np.zeros(3), np.zeros(3), 0

if __name__ == '__main__':
    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_pixels = 0

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image, image_files), total=len(image_files)))

    for mean, std, count in results:
        total_mean += mean
        total_std += std
        total_pixels += count

    if total_pixels > 0:
        mean = total_mean / total_pixels
        std = np.sqrt(total_std / total_pixels)

        print("Mean:", mean)
        print("Std:", std)
    else:
        print("No valid images processed.")
