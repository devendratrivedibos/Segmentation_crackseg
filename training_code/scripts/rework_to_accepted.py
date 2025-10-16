import os
import shutil
from glob import glob

def move_rework_to_accepted(rework_images_dir, rework_masks_dir,
                            accepted_images_dir, accepted_masks_dir,
                            only_matching_pairs=False):
    """
    Move images and masks from rework folders to accepted folders.

    Args:
        rework_images_dir (str): Path to rework images folder.
        rework_masks_dir (str): Path to rework masks folder.
        accepted_images_dir (str): Path to accepted images folder.
        accepted_masks_dir (str): Path to accepted masks folder.
        only_matching_pairs (bool): If True, move only files that exist in both images and masks.
    """

    os.makedirs(accepted_images_dir, exist_ok=True)
    os.makedirs(accepted_masks_dir, exist_ok=True)

    rework_images = {os.path.basename(p): p for p in glob(os.path.join(rework_images_dir, "*"))}
    rework_masks = {os.path.basename(p): p for p in glob(os.path.join(rework_masks_dir, "*"))}

    if only_matching_pairs:
        common_files = set(rework_images.keys()) & set(rework_masks.keys())
    else:
        common_files = set(rework_images.keys()) | set(rework_masks.keys())

    moved_count = 0
    for filename in common_files:
        if filename in rework_images:
            shutil.move(rework_images[filename], os.path.join(accepted_images_dir, filename))
        if filename in rework_masks:
            shutil.move(rework_masks[filename], os.path.join(accepted_masks_dir, filename))
        moved_count += 1

    print(f"✅ Moved {moved_count} file pairs from rework → accepted.")

# Example usage:
move_rework_to_accepted(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2\REWORK_IMAGES",
r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2\REWORK_MASKS",
r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2\ACCEPTED_IMAGES",
r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2\ACCEPTED_MASKS",
only_matching_pairs=True)
