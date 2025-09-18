import os
import shutil

# --- source folders ---
images_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_ASPHALT_OLD\AnnotationImages"
masks_dir  = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_ASPHALT_OLD\AnnotationMasks"

# --- destination root ---
dest_root = r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51"

# mapping of tag → section name
tag_to_section = {
    "A_T_3": "SECTION-3",
    "A_T_4": "SECTION-4",
    "A_T_5": "SECTION-5",
    # add more if needed
}

def copy_files(src_dir, subfolder):
    for filename in os.listdir(src_dir):
        for tag, section in tag_to_section.items():
            if tag in filename:
                dest_folder = os.path.join(dest_root, section, subfolder)
                os.makedirs(dest_folder, exist_ok=True)

                src_path  = os.path.join(src_dir, filename)
                dest_path = os.path.join(dest_folder, filename)

                shutil.copy2(src_path, dest_path)
                print(f"Copied {filename} -> {dest_path}")
                break  # stop checking after first match

# Copy images and masks
copy_files(images_dir, "AnnotationImages_A_T")
copy_files(masks_dir,  "AnnotationMasks_A_T")
