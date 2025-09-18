import os
import re

def process_section(section_path, name_prefix="DAMOH_"):
    process_distress = os.path.join(section_path, "process_distress")
    masks_dir = os.path.join(section_path, "Masks")

    if not os.path.exists(process_distress) or not os.path.exists(masks_dir):
        print("❌ Required folders missing.")
        return

    # Pattern to extract number from mask filename like: mask-0000049-_crack.png
    number_pattern = re.compile(r"mask-(\d+)-?_crack\.png", re.IGNORECASE)

    # Collect and sort mask files
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(".png")])

    for mask_file in mask_files:
        match = number_pattern.match(mask_file)
        if not match:
            print(f"⚠️ Skipping {mask_file}, pattern not matched.")
            continue

        number = match.group(1)
        image_file = f"prof-{number}-_crack.png"
        mask_path = os.path.join(masks_dir, mask_file)
        image_path = os.path.join(process_distress, image_file)

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found for {mask_file}, expected {image_file}, skipping.")
            continue

        # New names
        new_name = f"{name_prefix}{number}.png"
        img_out = os.path.join(process_distress, new_name)
        mask_out = os.path.join(masks_dir, new_name)

        # Rename instead of copy
        os.rename(image_path, img_out)
        os.rename(mask_path, mask_out)

        print(f"✔ Renamed pair: {mask_file} + {image_file} → {new_name}")


# process_section(
#     r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-2",
#     name_prefix='DAMOH_SIMARIYA_SECTION-2_'
# )
#
import os

# --- update paths here ---
image_dir = r"V:\CHAS-RAMGARH_2024-11-14_11-09-22\SECTION-3\AnnotationImages"
mask_dir  = r"V:\CHAS-RAMGARH_2024-11-14_11-09-22\SECTION-3\AnnotationMasks"

def rename_files(folder, old_prefix="C_S_1", new_prefix="C_S_3"):
    for filename in os.listdir(folder):
        if filename.startswith(old_prefix):
            old_path = os.path.join(folder, filename)
            new_filename = filename.replace(old_prefix, new_prefix, 1)
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
# Run for both folders
rename_files(image_dir)
rename_files(mask_dir)