import pandas as pd
import shutil
import os

# ---- CONFIG ----
ROOT_FOLDER = r"D:\LALITPUR-LAKHNADON_2025-11-16_10-35-41\SECTION-5"
# sanitize ROOT_FOLDER for prefix
# clean_root = ROOT_FOLDER.replace("\\", "_").replace(":", "")
clean_root = r"LALITPUR-LAKHNADON"
csv_potholes = rf"{ROOT_FOLDER}\reportNew\csv_reports\potholes.csv"
csv_cracks = rf"{ROOT_FOLDER}\process_distress\cracks_predictions.csv"

source_folders = {
    "overlayed_images": os.path.join(ROOT_FOLDER, "overlayed_images"),
    "pcams": os.path.join(ROOT_FOLDER, "pcams"),
    "process_distress": os.path.join(ROOT_FOLDER, "process_distress"),
}

output_base = os.path.join(ROOT_FOLDER, "output")
os.makedirs(output_base, exist_ok=True)

# ---- READ CSVs ----
df_potholes = pd.read_csv(csv_potholes)
df_cracks   = pd.read_csv(csv_cracks)
df_potholes = pd.read_csv(csv_potholes, on_bad_lines='skip')
df_cracks   = pd.read_csv(csv_cracks, on_bad_lines='skip')

# ---- FILENAME GENERATORS ----
def filename_overlay(region):
    return f"overlay_{region:04d}.png"

def filename_pcam(region):
    return f"pcam-{region:07d}-LL.jpg"

def filename_distress(region):
    return f"inference_input-{region:07d}-_crack.png"



# ------------------------------------------------------
#  PROCESS POTHOLES — USE SEVERITY
# ------------------------------------------------------
print("\n=== Processing potholes.csv (Severity) ===\n")

for _, row in df_potholes.iterrows():

    region = int(row['Region_index'])
    chainage = str(row['Chainage']).strip()
    severity = str(row['Severity']).strip()

    # Output folder: output/<severity>/
    target_folder = os.path.join(output_base, severity)
    os.makedirs(target_folder, exist_ok=True)

    filenames = {
        "overlayed_images": filename_overlay(region),
        "pcams": filename_pcam(region),
        "process_distress": filename_distress(region)
    }

    for folder, original_name in filenames.items():

        src = os.path.join(source_folders[folder], original_name)

        if os.path.exists(src):

            new_name = f"{clean_root}_RHS_INNER_{chainage}_{original_name}"
            dst = os.path.join(target_folder, new_name)

            shutil.copy(src, dst)
            print(f"Copied → {dst}")

        else:
            print(f"❌ Missing: {src}")


# ------------------------------------------------------
#  PROCESS CRACKS — USE CRACK_TYPE
# ------------------------------------------------------
print("\n=== Processing cracks.csv (Crack_type) ===\n")

for _, row in df_cracks.iterrows():

    region = int(row['Region_index'])
    chainage = str(row['Chainage']).strip()
    crack_type = str(row['crack_type']).strip()

    # Output folder: output/<crack_type>/
    target_folder = os.path.join(output_base, crack_type)
    os.makedirs(target_folder, exist_ok=True)

    filenames = {
        "overlayed_images": filename_overlay(region),
        "pcams": filename_pcam(region),
        "process_distress": filename_distress(region)
    }

    for folder, original_name in filenames.items():

        src = os.path.join(source_folders[folder], original_name)

        if os.path.exists(src):

            new_name = f"{clean_root}_RHS_INNER_{chainage}_{original_name}"
            dst = os.path.join(target_folder, new_name)

            shutil.copy(src, dst)
            print(f"Copied → {dst}")

        else:
            print(f"❌ Missing: {src}")
