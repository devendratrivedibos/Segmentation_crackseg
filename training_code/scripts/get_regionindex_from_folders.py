import os
import re

# Replace with your actual folder paths
folders = [
r"Y:\NSV_DATA\HAZARIBAGH-RANCHI_2024-10-07_11-25-27\SECTION-2\ACCEPTED_IMAGES",
]

region_indexes =[]
for folder in folders:
    for f in os.listdir(folder):
        if (f.lower().endswith(('.jpg', '.png', '.jpeg'))):
            match = re.search(r"IMG_(\d+)", f)
            # match = re.search(r"-(\d+)-_", f)
            if match:
                next_number = int(match.group(1))
                region_indexes.append(next_number)

print(sorted(set(region_indexes)))

