import os
import shutil

source_folder = r'W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress'      # Change this to your source folder
destination_folder = r'W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress'  # Change this to your destination folder
prefix = 'A_T_5_'                       # Change this to your desired prefix
# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)
# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
for filename in os.listdir(source_folder):
    if filename.lower().endswith(image_extensions):
        src = os.path.join(source_folder, filename)
        dst = os.path.join(destination_folder, prefix + filename)
        # os.rename(src, dst)
        # shutil.copy2(src, dst)
print('Images copied and renamed successfully.')

for f in os.listdir(source_folder):
    if f.startswith("A_T_1_A_T_q_rangeDataFiltered-") and f.endswith(".jpg"):
        new_name = f.replace("A_T_q_rangeDataFiltered-", "A_T_1_rangeDataFiltered-")
        old_path = os.path.join(source_folder, f)
        new_path = os.path.join(source_folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {f} -> {new_name}")