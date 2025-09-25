import os
import shutil

# Base path containing all SECTION folders
base_folder = r"z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51"

# Loop over all SECTION folders
for section_name in os.listdir(base_folder):
    section_path = os.path.join(base_folder, section_name)
    if not os.path.isdir(section_path) or not section_name.lower().startswith("section-4"):
        continue
    print(f"Handling section: {section_name}")

    process_distress = os.path.join(section_path, "process_distress_40")
    process_4030 = os.path.join(section_path, "IMAGES_4030")
    process_4040 = os.path.join(section_path, "IMAGES_4040")

    os.makedirs(process_4030, exist_ok=True)
    os.makedirs(process_4040, exist_ok=True)

    if not os.path.exists(process_distress):
        continue  # skip if process_distress doesn't exist

    for f in os.listdir(process_distress):
        src = os.path.join(process_distress, f)
        if not os.path.isfile(src):
            continue

        if "IMG_4030" in f:
            dst = os.path.join(process_4030, f)
            shutil.move(src, dst)
        elif "IMG_4040" in f:
            dst = os.path.join(process_4040, f)
            shutil.move(src, dst)

    print(f"Processed {section_name}")