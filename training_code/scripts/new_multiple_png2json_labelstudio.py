import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from label_studio_converter import brush
import glob

root_dir = r"C:\Users\Admin\Downloads\M"
newfolder = 57
username = 'Admin'

# Collect both PNG and JPG images (ignore JSON files)
image_files = []
for ext in ("*.png", "*.jpg", "*.jpeg"):
    image_files.extend(glob.glob(os.path.join(root_dir, ext)))

# Or you can hardcode for debugging
# image_files = [r"C:\Users\Admin\Downloads\M\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33_SECTION-1_IMG_0000000.png"]

json_path = fr"{root_dir}/_new_multiple_output_labelstudio.json"
prefix_folder = fr"C:/Users/{username}/AppData/Local/label-studio/label-studio/media/upload/{newfolder}"

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
    (100, 100, 100): "popout"  # Grey
}


def rgb_masks_to_labelstudio_json(image_paths, output_json,
                                  prefix_map,
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

        # Process each crack/pothole color
        for color, label in COLOR_TO_LABEL.items():
            class_mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1).astype(np.uint8) * 255
            if np.sum(class_mask) == 0:
                continue

            tmp_mask_path = f"{os.path.splitext(rgb_path)[0]}_{label}.jpg"
            Image.fromarray(class_mask, mode="L").save(tmp_mask_path)

            rle = brush.image2rle(tmp_mask_path)
            rle = rle[0]
            os.remove(tmp_mask_path)

            if not rle:  # skip empty masks
                continue

            results.append({
                "image_rotation": 0,
                "original_width": img.width,
                "original_height": img.height,
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

        # --- Match basename (ignore extension) ---
        base_name = os.path.basename(rgb_path)
        base_key = os.path.splitext(base_name)[0]  # without extension

        if base_key in prefix_map:
            rgb_name = prefix_map[base_key]  # use prefixed name with correct extension
        else:
            print(f"⚠️ Warning: {base_name} not found in prefix_map, skipping")
            continue

        # ✅ Always include annotation, even if results is empty
        annotation = {
            "id": start,
            "completed_by": completed_by,
            "result": results,   # may be []
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
            "file_upload": rgb_name,
            "drafts": [],
            "predictions": [],
            "data": {"image": f"/data/upload/{project_id}/{rgb_name}"},
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
    task["project"] = newfolder
    for ann in task.get("annotations", []):
        ann["project"] = newfolder
    task["data"]["image"] = f"/data/upload/{newfolder}/{task['file_upload']}"


# --- Build prefix_map (basename_noext -> prefixed_name with extension) ---
prefix_map = {}
for fname in os.listdir(prefix_folder):
    if fname.endswith((".png", ".jpg", ".jpeg")):
        parts = fname.split("-", 1)
        if len(parts) == 2:
            original_name = parts[1]
            key = os.path.splitext(original_name)[0]  # ignore extension
            prefix_map[key] = fname
print(prefix_map)

# --- Convert masks to JSON ---
rgb_masks_to_labelstudio_json(image_files, json_path, prefix_map)

# --- Load JSON & Update Tasks ---
with open(json_path, "r") as f:
    data = json.load(f)

if isinstance(data, list):
    for task in data:
        update_task(task, newfolder=newfolder)

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"🎯 Updated JSON saved → {json_path}")
