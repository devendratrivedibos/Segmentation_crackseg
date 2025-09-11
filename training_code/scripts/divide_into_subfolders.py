import os
import shutil
from pathlib import Path

def divide_into_subfolders(process_dir, results_dir, output_dir, n_subfolders=10):
    process_dir = Path(process_dir)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect base names (without extension)
    process_files = {f.stem: f for f in process_dir.iterdir() if f.is_file()}
    results_files = {f.stem: f for f in results_dir.iterdir() if f.is_file()}

    # Find matching base names
    matching_keys = sorted(set(process_files.keys()) & set(results_files.keys()))
    total = len(matching_keys)
    chunk_size = (total + n_subfolders - 1) // n_subfolders  # ceil division

    for i in range(n_subfolders):
        subfolder = output_dir / f"part_{i+1}"
        img_folder = subfolder / "img"
        mask_folder = subfolder / "mask"
        img_folder.mkdir(parents=True, exist_ok=True)
        mask_folder.mkdir(parents=True, exist_ok=True)

        for key in matching_keys[i*chunk_size:(i+1)*chunk_size]:
            # Copy from process_distress → img
            shutil.copy(process_files[key], img_folder / process_files[key].name)
            # Copy from sectionresults → mask
            shutil.copy(results_files[key], mask_folder / results_files[key].name)

    print(f"✅ Done! Divided {total} matching pairs into {n_subfolders} subfolders (img/mask).")

# Example usage:
process_dir = r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-1\process_distress"
results_dir = r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-1\Masks"
output_dir = r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-1\divided_images"
divide_into_subfolders(process_dir, results_dir, output_dir, n_subfolders=15)
process_dir = r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-2\process_distress"
results_dir = r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-2\Masks"
output_dir = r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-2\divided_images"

divide_into_subfolders(process_dir, results_dir, output_dir, n_subfolders=15)
