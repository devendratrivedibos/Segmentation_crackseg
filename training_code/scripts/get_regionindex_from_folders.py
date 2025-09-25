import os
import re

# Replace with your actual folder paths
folders = [
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5\ACCEPTED_MASKS",
]

region_indexes = []
for folder in folders:
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            # match = re.search(r"IMG_(\d+)", f)
            match = re.search(r"-(\d+)-_", f)
            if match:
                next_number = int(match.group(1))
                region_indexes.append(next_number)

print(sorted(region_indexes))

