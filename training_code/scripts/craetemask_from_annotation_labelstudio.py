import os
import pdb

import cv2
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict

# --- Paths ---
root_folder = r"X:\DataSet\25-08-2025\Sharan"
root_folder = r"C:\Users\Admin\Downloads"
image_folder = os.path.join(root_folder, "project-36-at-2025-08-30-17-26-e16eb6dd")
csv_file = os.path.join(root_folder, "project-36-at-2025-08-30-17-27-e16eb6dd.csv")
output_folder = os.path.join(root_folder, "Masks")
os.makedirs(output_folder, exist_ok=True)

# --- Step 1: Ensure all masks are binary ---
# for file in os.listdir(image_folder):
#     if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#         path = os.path.join(image_folder, file)
#         img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         _, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
#         cv2.imwrite(path, binary_img)
#
# print("All masks converted to binary.")

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
    pdb.set_trace()
    if 'alligator crack' in filename:
        return (255, 0, 0)   # Red
    elif 'transverse crack' in filename:
        return (0, 0, 255)   # Blue
    elif 'longitudinal crack' in filename:
        return (0, 255, 0)   # Green
    elif 'multiple crack' in filename:
        return (255, 0, 255) # Magenta
    elif 'pothole' in filename:
        return (0, 42, 255)  # Orange
    elif 'joint seal' in filename:
        return (255, 204, 0) # Yellow
    elif 'airplane' in filename:
        return (255, 0, 0) # Yellow
    elif 'car' in filename:
        return (0, 255, 0) # Yellow
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
