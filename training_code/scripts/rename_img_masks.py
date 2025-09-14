#@author: Devendra
'''
THIS 'copy_images_with_masks_folder' WILL COPY MASK AND IMAGES FROM process_distress AND Masks FOLDER
TO AnnotationImages AND AnnotationMasks FOLDER

process_section will copy prof-0000-crack mask and images after renaming into AnnotationImages and AnnotationMasks folder

'''


import os
import pdb
import re
import shutil

import os
import shutil

# Regex to extract number from filename
number_pattern = re.compile(r"(\d+)")

def process_section(section_path, name_prefix="THANE_BELAPUR_SECTION2_"):
    """
    RENAME and COPY images and masks from process_distress and Masks folders
    to AnnotationImages and AnnotationMasks with a new naming convention.
    """
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


def copy_matched_images(main_dir: str):
    """
    Copies images (jpg/png) from SECTION*/process_distress folders into AnnotationImages
    if their base names match any mask image in the Masks folder.
    loop through all sections/process distress
    masks should be outside section folder
    Ignores file extension differences.

    Args:
        main_dir (str): Path to the main directory containing SECTION folders and Masks.
    """
    masks_dir = os.path.join(main_dir, "AnnotationMasks")
    annotation_dir = os.path.join(main_dir, "AnnotationImages")

    # Create AnnotationImages folder if not exists
    os.makedirs(annotation_dir, exist_ok=True)

    # Collect mask base names (without extension)
    mask_files = {os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.lower().endswith(".png")}

    copied_count = 0
    valid_exts = {".png", ".jpg", ".jpeg"}

    # Iterate over SECTION folders
    for section in os.listdir(main_dir):
        section_path = os.path.join(main_dir, section)
        process_distress_path = os.path.join(section_path, "process_distress")

        if os.path.isdir(process_distress_path):
            for img_name in os.listdir(process_distress_path):
                base_name, ext = os.path.splitext(img_name)
                if base_name in mask_files and ext.lower() in valid_exts:
                    src = os.path.join(process_distress_path, img_name)
                    dst = os.path.join(annotation_dir, img_name)
                    try:
                        shutil.copy2(src, dst)
                        copied_count += 1
                        print(f"Copied: {img_name}")
                    except Exception as e:
                        print(f"⚠️ Skipped {img_name} ({e})")

    print(f"\n✅ Copying completed. Total images copied: {copied_count}")


def copy_images_with_masks_folder(section_path: str):
    """SEES MASKS FOLDER FOR ANNOTATION IMAGES
    COPY SAME NAME (JPG, PNG) FROM SECTION*/process_distress FOLDER
    1. For each SECTION folder, look for process_distress and AnnotationMasks folders
    2. Copy images from process_distress to AnnotationImages if their base names match any mask in AnnotationMasks
    3. Ignore file extension differences (jpg/png)
    """

    process_distress = os.path.join(section_path, "process_distress")
    masks_folder = os.path.join(section_path, "AnnotationMasks")
    annotation_images = os.path.join(section_path, "AnnotationImages")

    os.makedirs(annotation_images, exist_ok=True)

    # Collect mask basenames (ignoring extension)
    mask_basenames = {
        os.path.splitext(f)[0] for f in os.listdir(masks_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }

    # Iterate process_distress images
    for img_file in os.listdir(process_distress):
        if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_base, ext = os.path.splitext(img_file)
        if img_base in mask_basenames:
            src = os.path.join(process_distress, img_file)
            dst = os.path.join(annotation_images, img_file)
            shutil.copy2(src, dst)  # copy with metadata

    print(f"Processed section: {section_path}")


# Example usage
if __name__ == "__main__":
    main_folder = r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33"
    # copy_matched_images(main_folder)


    # copy_images_with_masks_folder(r"Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-1")
    # copy_images_with_masks_folder(r"Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-3")
    # copy_images_with_masks_folder(r"Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-4")

    # copy_images_with_masks_folder(r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-1")
    # copy_images_with_masks_folder(r"W:\BOS\DAMOH-SIMARIYA_2025-06-17_05-55-01\SECTION-2")

    copy_images_with_masks_folder(r"V:\KHARWANDIKASAR-PADALSHINGI_2024-12-21_12-15-00\SECTION-1")
    # copy_images_with_masks_folder(r"V:\KHARWANDIKASAR-PADALSHINGI_2024-12-21_12-15-00\SECTION-2")
    # copy_images_with_masks_folder(r"V:\KHARWANDIKASAR-PADALSHINGI_2024-12-21_12-15-00\SECTION-3")

    # copy_images_with_masks_folder(r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1")
    # copy_images_with_masks_folder(r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-2")
    # copy_images_with_masks_folder(r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-3")
    # copy_images_with_masks_folder(r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-4")
    # copy_images_with_masks_folder(r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5")
    # copy_images_with_masks_folder(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-7")
    # copy_images_with_masks_folder(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-7")

    # # Base projects folder
    # process_section(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3", name_prefix="THANE_BELAPUR_SECTION3_")
    # process_section(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3", name_prefix="THANE_BELAPUR_SECTION3_")
    # process_section(r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2", name_prefix="THANE_BELAPUR_SECTION2_")