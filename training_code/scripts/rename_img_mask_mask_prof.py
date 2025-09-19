import os
import re
import shutil

def process_section(section_path, name_prefix="DAMOH_"):
    process_distress = os.path.join(section_path, "process_distress")
    masks_dir = os.path.join(section_path, "Masks")

    if not os.path.exists(process_distress):
        print("❌ process_distress folder missing.")
        return

    os.makedirs(masks_dir, exist_ok=True)

    # Pattern to extract number from mask filename like: mask-0000049-_crack.png
    number_pattern = re.compile(r"mask-(\d+)-?_crack\.png", re.IGNORECASE)

    # Collect mask files from process_distress itself
    mask_files = sorted([
        f for f in os.listdir(process_distress)
        if f.lower().endswith(".png") and f.lower().startswith("mask-")
    ])

    for mask_file in mask_files:
        match = number_pattern.match(mask_file)
        if not match:
            print(f"⚠️ Skipping {mask_file}, pattern not matched.")
            continue

        number = match.group(1)
        image_file = f"prof-{number}-_crack.png"
        mask_path = os.path.join(process_distress, mask_file)
        image_path = os.path.join(process_distress, image_file)

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found for {mask_file}, expected {image_file}, skipping.")
            continue

        # New name (same for img + mask, different folders)
        new_name = f"{name_prefix}{number}.png"

        img_out = os.path.join(process_distress, new_name)  # stays here
        mask_out = os.path.join(masks_dir, new_name)        # moved to Masks/

        # Rename image (prof)
        os.rename(image_path, img_out)

        # Move + Rename mask
        shutil.move(mask_path, mask_out)

        print(f"✔ Renamed + moved pair → {new_name}")

# Example usage

process_section(r"V:\SANKHA-KHAJURI_2024-11-12_13-30-38\SECTION-3",
    name_prefix='SANKHA-KHAJURI_SECTION-3_')
process_section(r"V:\SANKHA-KHAJURI_2024-11-12_13-30-38\SECTION-4",
    name_prefix='SANKHA-KHAJURI_SECTION-4_')
process_section(r"V:\SANKHA-KHAJURI_2024-11-12_13-30-38\SECTION-5",
    name_prefix='SANKHA-KHAJURI_SECTION-5_')
process_section(r"V:\SANKHA-KHAJURI_2024-11-12_13-30-38\SECTION-6",
    name_prefix='SANKHA-KHAJURI_SECTION-6_')
process_section(r"V:\SANKHA-KHAJURI_2024-11-12_13-30-38\SECTION-7",
    name_prefix='SANKHA-KHAJURI_SECTION-7_')
process_section(r"V:\SANKHA-KHAJURI_2024-11-12_13-30-38\SECTION-8",
    name_prefix='SANKHA-KHAJURI_SECTION-8_')
process_section(r"V:\SANKHA-KHAJURI_2024-11-12_13-30-38\SECTION-9",
    name_prefix='SANKHA-KHAJURI_SECTION-9_')
process_section(r"V:\SANKHA-KHAJURI_2024-11-12_13-30-38\SECTION-10",
    name_prefix='SANKHA-KHAJURI_SECTION-10_')

#
# import os
#
# # --- update paths here ---
# image_dir = r"V:\CHAS-RAMGARH_2024-11-14_11-09-22\SECTION-3\AnnotationImages"
# mask_dir  = r"V:\CHAS-RAMGARH_2024-11-14_11-09-22\SECTION-3\AnnotationMasks"
#
# def rename_files(folder, old_prefix="C_S_1", new_prefix="C_S_3"):
#     for filename in os.listdir(folder):
#         if filename.startswith(old_prefix):
#             old_path = os.path.join(folder, filename)
#             new_filename = filename.replace(old_prefix, new_prefix, 1)
#             new_path = os.path.join(folder, new_filename)
#             os.rename(old_path, new_path)
#             print(f"Renamed: {filename} -> {new_filename}")
# # Run for both folders
# rename_files(image_dir)
# rename_files(mask_dir)