import os
import re
import shutil

# Base projects folder
projects_dir = r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2"

# Regex to extract number from filename
number_pattern = re.compile(r"(\d+)")

def process_section(section_path, name_prefix="THANE_BELAPUR_SECTION2_"):
    process_distress = os.path.join(section_path, "process_distress")
    masks_dir = os.path.join(section_path, "Masks")

    if not os.path.exists(process_distress) or not os.path.exists(masks_dir):
        return

    # Output folders
    ann_images_dir = os.path.join(section_path, "AnnotationImages")
    ann_masks_dir = os.path.join(section_path, "AnnotationMasks")
    os.makedirs(ann_images_dir, exist_ok=True)
    os.makedirs(ann_masks_dir, exist_ok=True)

    # Collect masks available
    mask_files = {f for f in os.listdir(masks_dir) if f.lower().endswith((".jpg", ".png"))}

    # Process each mask (since we only keep pairs)
    for mask_file in sorted(mask_files):
        # Extract number from mask filename
        match = number_pattern.search(mask_file)
        if not match:
            print(f"⚠️ Skipping {mask_file}, no number found")
            continue

        number = match.group(1).zfill(6)
        new_name = f"{name_prefix}{number}.png"

        # Paths
        mask_path = os.path.join(masks_dir, mask_file)
        img_path = os.path.join(process_distress, mask_file)  # assume same name

        # Destination
        mask_out = os.path.join(ann_masks_dir, new_name)
        img_out = os.path.join(ann_images_dir, new_name)

        # Only copy if corresponding image exists
        if os.path.exists(img_path):
            shutil.copy(img_path, img_out)
            shutil.copy(mask_path, mask_out)
            print(f"✔ Copied pair: {mask_file} → {new_name}")
        else:
            print(f"⚠️ No matching image for {mask_file}, skipped")


if __name__ == "__main__":
    # process_section(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2", name_prefix="THANE_BELAPUR_SECTION2_")
    # process_section(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3", name_prefix="THANE_BELAPUR_SECTION3_")
    process_section(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-4", name_prefix="THANE_BELAPUR_SECTION4_")