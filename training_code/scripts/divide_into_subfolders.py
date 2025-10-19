import os
import shutil
from pathlib import Path


def divide_into_subfolders(process_dir, results_dir, output_dir, n_subfolders=10):
    process_dir = Path(process_dir)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Allow only image extensions
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Collect base names (without extension), ignoring Thumbs.db or non-images
    process_files = {
        f.stem: f for f in process_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_ext
    }
    results_files = {
        f.stem: f for f in results_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_ext
    }

    # Find matching base names
    matching_keys = sorted(set(process_files.keys()) & set(results_files.keys()))
    total = len(matching_keys)
    chunk_size = (total + n_subfolders - 1) // n_subfolders  # ceil division
    part = 1
    for i in range(n_subfolders):
        # Separate folders for images and masks
        output_dir = Path(output_dir)

        img_folder = output_dir/ f"{part}" / f"AnnotationImages{i+1}"
        mask_folder = output_dir/f"{part}" / f"AnnotationMasks{i+1}"
        img_folder.mkdir(parents=True, exist_ok=True)
        mask_folder.mkdir(parents=True, exist_ok=True)
        part += 1
        for key in matching_keys[i*chunk_size:(i+1)*chunk_size]:
            shutil.copy(process_files[key], img_folder / process_files[key].name)
            shutil.copy(results_files[key], mask_folder / results_files[key].name)

    print(f"✅ Done! Divided {total} matching pairs into {n_subfolders} subfolders (img/mask).")


process_dir = r"W:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\REWORK_IMAGES"
results_dir = r"W:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\REWORK_MASKS"
output_dir = r"W:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\REWORK_divided_images"
divide_into_subfolders(process_dir, results_dir, output_dir, n_subfolders=5)
