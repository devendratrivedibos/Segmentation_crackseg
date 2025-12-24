import os
import re
import csv

# Folder containing files
FOLDER_PATH = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\ACCEPTED_MASKS"
OUTPUT_CSV = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\ACCEPTED_MASKS\asphalt.csv"

results = []

pattern = re.compile(
    r"(.*?)_(SECTION-\d+)_IMG_(\d+)"
)

for filename in os.listdir(FOLDER_PATH):
    name, _ = os.path.splitext(filename)

    match = pattern.match(name)
    if match:
        project_name = match.group(1)
        section = match.group(2)
        image_index = int(match.group(3))  # remove leading zeros

        results.append([
            project_name,
            section,
            image_index
        ])

# Write to CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["project_name", "section", "image_index"])
    writer.writerows(results)

print(f"Saved {len(results)} records to {OUTPUT_CSV}")
