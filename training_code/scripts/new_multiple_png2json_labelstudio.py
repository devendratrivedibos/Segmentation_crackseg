import os
import json
import pdb
import numpy as np
from PIL import Image
from datetime import datetime
from label_studio_converter import brush
import glob

image_files = glob.glob(r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_CLASSMasks\*.png")
json_path = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_CLASSMasks\_new_multiple_output_labelstudio.json"
prefix_folder = r"C:\Users\Admin\AppData\Local\label-studio\label-studio\media\upload\50"  # folder containing prefixed PNG files
newfolder = 50


COLOR_TO_LABEL = {
    (255, 0, 0): "Alligator crack",
    (0, 255, 0): "Longitudinal crack",
    (0, 0, 255): "Transverse crack",
    (139, 69, 19): "pothole",  # Brown
    (255, 165, 0): "patch",  # Orange
    (128, 0, 128): "punchout",  # Purple
    (0, 255, 255): "spalling",  # Cyan
    (0, 128, 0): "corner break",  # Dark green
    (255, 100, 203): "joint sealed transverse crack",  # Light pink
    (199, 21, 133): "joint sealed longitudinal crack",  # Dark pink
    (255, 215, 0): "cracking",  # Gold
    (255, 255, 255): "unclassified",  # White
    (255, 0, 255): "multiple crack",  # Yellow
    (100,100,100): "popout"    # Grey
}


def print_structure(data, indent=0):
    """Recursively print JSON structure (keys nesting)."""
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}- {key}")
            print_structure(value, indent + 1)
    elif isinstance(data, list):
        print(f"{prefix}- [list]")
        if len(data) > 0:
            print_structure(data[0], indent + 1)  # show structure of first element only


def rgb_masks_to_labelstudio_json(image_paths, output_json,
                                  project_id=42, from_name="tag",
                                  to_name="image", completed_by=1,
                                  user_id=1):
    all_tasks = []
    now = datetime.utcnow().isoformat() + "Z"

    for task_id, rgb_path in enumerate(image_paths, start=1):

        img = Image.open(rgb_path).convert("RGB")
        rgb = np.array(img)
        results = []
        start = 1
        for color, label in COLOR_TO_LABEL.items():
            class_mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1).astype(np.uint8) * 255
            if np.sum(class_mask) == 0:
                continue
            tmp_mask_path = f"{os.path.splitext(rgb_path)[0]}_{label}.png"
            Image.fromarray(class_mask, mode="L").save(tmp_mask_path)

            rle = brush.image2rle(tmp_mask_path)
            rle = rle[0]
            os.remove(tmp_mask_path)

            if not rle:  # skip empty masks
                continue

            results.append({
                "image_rotation": 0,
                "original_width": 419,
                "original_height": 1024,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [label]
                },
                "id": os.urandom(6).hex(),
                "from_name": from_name,
                "to_name": to_name,
                "type": "brushlabels",
                "origin": "manual"
            })

        if not results:  # skip image if no masks
            continue

        annotation = {
            "id": start,
            "completed_by": completed_by,
            "result": results,
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": now,
            "updated_at": now,
            "lead_time": 0,
            "prediction": {},
            "result_count": len(results),
            "unique_id": os.urandom(8).hex(),
            "task": task_id,
            "project": project_id,
            "updated_by": user_id,
        }

        task = {
            "id": task_id,
            "annotations": [annotation],
            "file_upload": os.path.basename(rgb_path),
            "drafts": [],
            "predictions": [],
            "data": {"image": f"/data/upload/{project_id}/{os.path.basename(rgb_path)}"},
            "meta": {},
            "created_at": now,
            "updated_at": now,
            "inner_id": task_id,
            "total_annotations": 1,
            "project": project_id,
            "updated_by": user_id,
        }

        all_tasks.append(task)
        start += 1
    with open(output_json, "w") as f:
        json.dump(all_tasks, f, indent=2)

    print(f"✅ Saved Label Studio JSON for {len(all_tasks)} images → {output_json}")


def update_task(task, newfolder):
    old_file = task.get("file_upload")
    if not old_file:
        return

    # Find matching prefix file by suffix
    new_file = prefix_map.get(old_file)
    if new_file:
        # Update file_upload
        task["file_upload"] = new_file

        # Update project field at task level
        task["project"] = newfolder

        # Update project inside annotations
        if "annotations" in task:
            for ann in task["annotations"]:
                ann["project"] = newfolder

        # Update data.image if exists
        if "data" in task and "image" in task["data"]:
            old_path = task["data"]["image"]

            # Extract current upload folder (e.g. /data/upload/42)
            old_folder = os.path.dirname(old_path)

            # Replace the old upload folder number with the new one
            parts = old_folder.split("/")
            if len(parts) >= 3 and parts[-2] == "upload":
                parts[-1] = str(newfolder)  # swap folder number
            new_folder_path = "/".join(parts)

            # Join with new file name
            task["data"]["image"] = os.path.join(new_folder_path, new_file)


rgb_masks_to_labelstudio_json(image_files, json_path)

# # Load JSON file
# with open(r"C:\Users\Admin\Downloads\new_multiple_output_labelstudio.json", "r") as f:
#     json_data = json.load(f)
# # Print structure
# print_structure(json_data)


# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)  # data is a list of tasks

    # Build a mapping: suffix -> prefixed_filename
    prefix_map = {}
    for fname in os.listdir(prefix_folder):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            parts = fname.split("-", 1)  # split only on the first dash
            if len(parts) == 2:
                suffix = parts[1]  # e.g. A_T_1_rangeDataFiltered-0000022-_crack.png
                prefix_map[suffix] = fname

# # Update each task in the list
# if isinstance(data, list):
#     for task in data:
#         update_task(task, newfolder=newfolder)
#
# # Save updated JSON
# with open(json_path, "w") as f:
#     json.dump(data, f, indent=2)
