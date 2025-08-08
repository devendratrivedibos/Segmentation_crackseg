from pathlib import Path
from PIL import Image
import numpy as np

def combine_all_masks(folder1, folder2, output_folder):
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all unique filenames from both folders
    filenames1 = set(f.name for f in folder1_path.glob('*'))
    filenames2 = set(f.name for f in folder2_path.glob('*'))
    all_filenames = filenames1.union(filenames2)

    for name in all_filenames:
        mask1_path = folder1_path / name
        mask2_path = folder2_path / name
        combined_mask_path = output_path / name

        if mask1_path.exists() and mask2_path.exists():
            # Load both masks
            mask1 = Image.open(mask1_path).convert('RGB')
            mask2 = Image.open(mask2_path).convert('RGB')

            arr1 = np.array(mask1)
            arr2 = np.array(mask2)
            combined = np.zeros_like(arr1)

            mask1_non_black = np.any(arr1 != [0, 0, 0], axis=-1)
            mask2_non_black = np.any(arr2 != [0, 0, 0], axis=-1)

            combined[mask1_non_black] = arr1[mask1_non_black]
            combined[mask2_non_black] = arr2[mask2_non_black]

            combined_img = Image.fromarray(combined)
            combined_img.save(combined_mask_path)
        elif mask1_path.exists():
            # Copy mask1 if only it exists
            mask1 = Image.open(mask1_path)
            mask1.save(combined_mask_path)
        elif mask2_path.exists():
            # Copy mask2 if only it exists
            mask2 = Image.open(mask2_path)
            mask2.save(combined_mask_path)

    return f'Combined masks saved to {output_folder}'


combine_all_masks(r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Vishwas\6.8.2025\5911\Masks',
                    r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Vishwas\6.8.2025\SAm\Masks',
                    r'D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Vishwas\Masks')
