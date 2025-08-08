import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import os

root_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Devendra"
image_folder = os.path.join(root_folder, "project-11-at-2025-08-08-20-27-d5d12b6f")
csv_file = os.path.join(root_folder, "project-11-at-2025-08-08-20-27-d5d12b6f.csv")

output_folder = os.path.join(root_folder, "Masks")
os.makedirs(output_folder, exist_ok=True)
# --- Step 1: Read the CSV and build the id-to-image-name mapping ---

df = pd.read_csv(csv_file)
df = df.dropna(subset=['id', 'image'])

# Extract relevant portion for saving file
df['image_name'] = df['image'].apply(lambda x: x.split('/')[-1])
df['processed_image_name'] = df['image_name'].apply(lambda x: x.split('-', 1)[1] if '-' in x else x)
id_to_image_name = dict(zip(df['id'], df['processed_image_name']))

# --- Step 2: List files in your annotation directory ---
# image_folder = r"C:\Users\Admin\Downloads\project-11-at-2025-08-04-19-45-ad0781aa"
list_dir = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# --- Step 3: Build the mapping of task mask(s) to desired image name ---
result_mapping = {}
for img_file in list_dir:
    prefix = img_file.split('-annotation')[0]
    try:
        task_num = int(prefix.replace('task-', ''))
    except:
        continue
    if task_num in id_to_image_name:
        result_mapping[img_file] = id_to_image_name[task_num]
    else:
        result_mapping[img_file] = img_file  # fallback if not in CSV

# --- Step 4: Group image files by task id ---
files_by_task = defaultdict(list)
for img_file in list_dir:
    task_prefix = img_file.split('-annotation')[0]
    files_by_task[task_prefix].append(img_file)


# --- Step 5: Define coloring logic for mask overlays ---
def get_mask_color(filename):
    filename = filename.lower()  # Convert to lowercase for case-insensitive matching
    if 'alligator crack' in filename:
        return (255, 0, 0)  # Red
    elif 'transverse crack' in filename:
        return (0, 0, 255)  # Green
    elif 'longitudinal crack' in filename:
        return (0, 255, 0)  # Blue
    elif 'multiple crack' in filename:
        return (255, 0, 255)  # Magenta
    elif 'pothole' in filename:
        return (0, 42, 255)  # Orange
    elif 'joint seal' in filename:
        return (255, 204, 0)  # Yellow
    else:
        print(f"FILENAME {filename} → NO MATCHING COLOR")
        return None


def combine_mask_images_color(image_files, folder):
    base_img = None
    for img_file in image_files:
        color = get_mask_color(img_file)
        if color is None:
            continue
        img_path = os.path.join(folder, img_file)
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        color_mask = np.zeros((*img_array.shape, 3), dtype=np.uint8)
        mask_nonzero = img_array > 0
        for i in range(3):
            color_mask[..., i][mask_nonzero] = color[i]
        if base_img is None:
            base_img = color_mask
        else:
            base_img = np.maximum(base_img, color_mask)
    if base_img is not None:
        return Image.fromarray(base_img)
    else:
        return None


# --- Step 6: Merge, color, and save each set of annotation masks ---
for task_prefix, files in files_by_task.items():
    combined_mask = combine_mask_images_color(files, image_folder)
    if combined_mask:
        save_name = result_mapping.get(files[0], files[0])  # get mapped save name
        save_path = os.path.join(output_folder, save_name)
        try:
            combined_mask.save(save_path)
            print(f"Saved combined mask as {save_name}")
        except:
            print(f"Failed to save combined mask for {save_name} {save_path}. Check file permissions or path validity.")
print("All combined colored masks have been saved with proper CSV names.")
