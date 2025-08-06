import os
import shutil

source_folder = r'D:\cracks\3Channel\Semantic-Segmentation of pavement distress dataset\AMRAVTI-TALEGAON_2025-06-09_06-38-51\SECTION-5\process_distress'      # Change this to your source folder
destination_folder = r'D:\cracks\3Channel\Semantic-Segmentation of pavement distress dataset\Combined'  # Change this to your destination folder
prefix = 'A_T_5_'                       # Change this to your desired prefix

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

for filename in os.listdir(source_folder):
    if filename.lower().endswith(image_extensions):
        src = os.path.join(source_folder, filename)
        dst = os.path.join(destination_folder, prefix + filename)
        shutil.copy2(src, dst)

print('Images copied and renamed successfully.')
