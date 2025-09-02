import os
import json
import pdb

# Paths
json_path = r"C:\Users\Admin\Downloads\multiple_output_labelstudio.json"        # your JSON file
prefix_folder = r"C:\Users\Admin\AppData\Local\label-studio\label-studio\media\upload\42"        # folder containing prefixed PNG files

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)  # data is a list of tasks

# Build a mapping: suffix -> prefixed_filename
prefix_map = {}
for fname in os.listdir(prefix_folder):
    if fname.endswith(".png") or fname.endswith(".jpg"):
        parts = fname.split("-", 1)  # split only on the first dash
        if len(parts) == 2:
            suffix = parts[1]        # e.g. A_T_1_rangeDataFiltered-0000022-_crack.png
            prefix_map[suffix] = fname

def update_task(task):
    old_file = task.get("file_upload")
    if not old_file:
        return

    # Find matching prefix file by suffix
    new_file = prefix_map.get(old_file)
    if new_file:
        # Update file_upload
        task["file_upload"] = new_file

        # Update data.image if exists
        if "data" in task and "image" in task["data"]:
            old_path = task["data"]["image"]
            folder = os.path.dirname(old_path)  # keep same upload folder
            task["data"]["image"] = os.path.join(folder, new_file)

# Update each task in the list
if isinstance(data, list):
    for task in data:
        update_task(task)

# Save updated JSON
with open(r"C:\Users\Admin\Downloads\new_multiple_output_labelstudio.json", "w") as f:
    json.dump(data, f, indent=2)
