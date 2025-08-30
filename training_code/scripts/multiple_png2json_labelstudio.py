import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from label_studio_converter import brush
import glob

COLOR_TO_LABEL = {
    (255, 0, 0): "Airplane",
    (0, 255, 0): "Car",
    (0, 0, 255): "Airplane",
    (0, 255, 0): "Car",
}

def rgb_masks_to_labelstudio_json(image_paths, output_json,
                                  project_id=33, from_name="tag",
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



# Example usage:
image_files = [
    r"C:\Users\Admin\Downloads\Masks\1A_T_1_rangeDataFiltered-0000025-_crack.png",
    r"C:\Users\Admin\Downloads\Masks\1A_T_1_rangeDataFiltered-0000026-_crack.png"
]

image_files = glob.glob(r"C:\Users\Admin\Downloads\Masks\*.png")

rgb_masks_to_labelstudio_json(image_files,
                              r"C:\Users\Admin\Downloads\multiple_output_labelstudio.json")
