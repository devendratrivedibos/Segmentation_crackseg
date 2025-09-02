import os
import pdb

import cv2
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict

# --- Paths ---
root_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Vishwas\New"
image_folder = os.path.join(root_folder, "project-16-at-2025-08-30-11-08-3b6b5eae")
csv_file = os.path.join(root_folder, "project-16-at-2025-08-30-11-08-3b6b5eae.csv")
output_folder = os.path.join(root_folder, "Masks")
os.makedirs(output_folder, exist_ok=True)


# --- Step 2: Read CSV mapping ---
df = pd.read_csv(csv_file)
df = df.dropna(subset=['id', 'image'])
df['image_name'] = df['image'].apply(lambda x: x.split('/')[-1])
df['processed_image_name'] = df['image_name'].apply(lambda x: x.split('-', 1)[1] if '-' in x else x)
id_to_image_name = dict(zip(df['id'], df['processed_image_name']))

# --- Step 3: Group files by task ---
list_dir = [f for f in os.listdir(image_folder) if f.endswith('.png')]
files_by_task = defaultdict(list)
result_mapping = {}

for img_file in list_dir:
    prefix = img_file.split('-annotation')[0]
    try:
        task_num = int(prefix.replace('task-', ''))
    except:
        continue
    files_by_task[prefix].append(img_file)
    result_mapping[img_file] = id_to_image_name.get(task_num, img_file)

# --- Step 4: Color mapping ---
def get_mask_color(filename):
    filename = filename.lower()

    if 'alligator crack' in filename:
        return (255, 0, 0)   # Red
    elif 'transverse crack' in filename:
        return (0, 0, 255)   # Blue
    elif 'longitudinal crack' in filename:
        return (0, 255, 0)   # Green
    elif 'multiple crack' in filename:
        return (255, 0, 255) # Magenta
    elif 'pothole' in filename:
        return (139, 69, 19)  # Orange
    elif "patch" in filename:
        return (255, 165, 0)

    elif "punchout" in filename:
        return (128, 0, 128)

    elif "spalling" in filename:
        return (0, 255, 255)

    elif "corner break" in filename:
         return (0, 128, 0)

    elif "Sealed Joint – T" in filename:
        return (255, 192, 203)

    elif "Sealed Joint – L" in filename:
        return (199, 21, 133)

    if "cracking" in filename:
        return (255, 215, 0)
    else:
        return None

# --- Step 5: Combine masks for each task ---
def combine_mask_images_color(image_files, folder):
    base_img = None
    for img_file in image_files:
        color = get_mask_color(img_file)
        if color is None:
            print("None Colors:", img_file)
            continue

        img_path = os.path.join(folder, img_file)
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        # Binary threshold to avoid stray values
        mask_nonzero = img_array > 0

        if base_img is None:
            base_img = np.zeros((*img_array.shape, 3), dtype=np.uint8)

        base_img[mask_nonzero] = color  # Overwrite where mask is present

    return Image.fromarray(base_img) if base_img is not None else None


# --- Step 6: Save results ---
for task_prefix, files in files_by_task.items():
    combined_mask = combine_mask_images_color(files, image_folder)
    if combined_mask:
        save_name = result_mapping.get(files[0], files[0])
        # Force PNG extension
        base_name, _ = os.path.splitext(save_name)
        save_name = base_name + ".png"
        save_path = os.path.join(output_folder, save_name)
        combined_mask.save(save_path, format='PNG')  # No interpolation
        print(f"Saved: {save_name}")

print("All combined masks saved successfully.")
