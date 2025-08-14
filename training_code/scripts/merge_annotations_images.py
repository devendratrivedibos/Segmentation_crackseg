from pathlib import Path
from PIL import Image
import numpy as np
from pathlib import Path
from PIL import Image
import numpy as np

def combine_all_masks(folder1, folder2, output_folder):
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {'.png', '.jpg', '.jpeg'}
    filenames1 = {f.name for f in folder1_path.iterdir() if f.suffix.lower() in valid_exts}
    filenames2 = {f.name for f in folder2_path.iterdir() if f.suffix.lower() in valid_exts}
    all_filenames = filenames1.union(filenames2)

    for name in all_filenames:
        mask1_path = folder1_path / name
        mask2_path = folder2_path / name

        arr1 = arr2 = None
        if mask1_path.exists():
            arr1 = np.array(Image.open(mask1_path).convert('RGB'))
        if mask2_path.exists():
            arr2 = np.array(Image.open(mask2_path).convert('RGB'))

        if arr1 is None and arr2 is None:
            continue

        if arr1 is None:
            combined = arr2
        elif arr2 is None:
            combined = arr1
        else:
            # Start with mask1
            combined = arr1.copy()
            mask2_non_black = np.any(arr2 != [0, 0, 0], axis=-1)
            mask1_black = np.all(combined == [0, 0, 0], axis=-1)
            # Fill in mask2 pixels only where mask1 is black
            combined[mask2_non_black & mask1_black] = arr2[mask2_non_black & mask1_black]

        # Always save as PNG
        save_name = Path(name).stem + ".png"
        Image.fromarray(combined.astype(np.uint8)).save(output_path / save_name)

    return f'Combined masks saved to {output_folder}'


combine_all_masks(r'X:\DataSet\1\BUSHRA_11-08-2025\Masks',
                    r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\segmentation_dataset_08_aug\Masks',
                    r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\segmentation_dataset_08_aug\Masks\out')


"""
import os
import numpy as np
from PIL import Image

# Paths to the two input folders and output folder
folder1 = r'X:\DataSet\1\BUSHRA_11-08-2025\Masks'
folder2 = r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\segmentation_dataset_08_aug\Masks'
output_folder = r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\segmentation_dataset_08_aug\Masks\out'
os.makedirs(output_folder, exist_ok=True)

# Define allowed colors (RGB tuples)
allowed_colors = [
            (0, 0, 0),       # Black
            (255, 0, 0),  # Green
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 0, 255),   # Yellow
            (255, 204, 0),  # Yellow  - Joint Seal
            (0, 42, 255),  # Orange  - Pothole
            (255, 255, 255)
]

# List of common image files in both folders to combine
files_folder1 = set(f for f in os.listdir(folder1) if f.lower().endswith('.jpg'))
files_folder2 = set(f for f in os.listdir(folder2) if f.lower().endswith('.jpg'))
common_files = files_folder1.intersection(files_folder2)

for filename in common_files:
    img1_path = os.path.join(folder1, filename)
    img2_path = os.path.join(folder2, filename)

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Prepare output array
    combined_arr = np.zeros_like(arr1)

    # Create mask where img1 pixels are non-black
    mask_img1_non_black = np.any(arr1 != [0, 0, 0], axis=-1)

    # For those pixels, take from img1
    combined_arr[mask_img1_non_black] = arr1[mask_img1_non_black]

    # For other pixels, take from img2
    combined_arr[~mask_img1_non_black] = arr2[~mask_img1_non_black]

    # Optional: Clamp colors to allowed_colors (in case img2 or img1 have other shades)
    def clamp_to_allowed_colors(image_array, allowed_colors):
        # For each pixel, set to closest allowed color (simple Euclidean distance)
        shape = image_array.shape
        flat_pixels = image_array.reshape(-1, 3)
        allowed = np.array(allowed_colors)

        # Compute distances and find nearest allowed color indices
        dists = np.sqrt(((flat_pixels[:, None, :] - allowed[None, :, :]) ** 2).sum(axis=2))
        nearest_indices = dists.argmin(axis=1)
        clipped_pixels = allowed[nearest_indices]

        return clipped_pixels.reshape(shape)

    combined_arr = clamp_to_allowed_colors(combined_arr, allowed_colors)

    # Save combined image
    combined_img = Image.fromarray(combined_arr)
    combined_img.save(os.path.join(output_folder, filename))
    print(f"Saved combined mask: {filename}")

print("All masks combined successfully with allowed colors only.")
"""