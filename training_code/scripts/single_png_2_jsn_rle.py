import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from pycocotools import mask as mask_utils
from label_studio_converter import brush


COLOR_TO_LABEL = {
    (255, 0, 0): "Airplane",
    (0, 255, 0): "Car",
}

# Encode mask as proper RLE
def encode_mask_to_rle(mask_path):
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # bytes → string for JSON
    return rle


def rgb_mask_to_full_json(rgb_path, output_json,
                          task_id=1, ann_id=79, project_id=33,
                          from_name="tag", to_name="image",
                          completed_by=1, user_id=1):

    # Load image
    img = Image.open(rgb_path).convert("RGB")
    rgb = np.array(img)
    height, width = rgb.shape[:2]

    results = []
    for color, label in COLOR_TO_LABEL.items():
        # Binary mask for this label
        class_mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1).astype(np.uint8) * 255
        tmp_mask_path = f"{os.path.splitext(rgb_path)[0]}_{label}.png"
        Image.fromarray(class_mask, mode="L").save(tmp_mask_path)

        # Encode with LS converter
        rle = brush.image2rle(tmp_mask_path)
        rle = rle[0]
        # rle = encode_mask_to_rle(tmp_mask_path)
        result = {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": [label]
            },
            "id": os.urandom(6).hex(),  # random UID for each annotation result
            "from_name": from_name,
            "to_name": to_name,
            "type": "brushlabels",
            "origin": "manual"
        }
        results.append(result)
        os.remove(tmp_mask_path)

    # ISO timestamp
    now = datetime.utcnow().isoformat() + "Z"

    # Annotation object
    annotation = {
        "id": ann_id,
        "completed_by": completed_by,
        "result": results,
        "was_cancelled": False,
        "ground_truth": False,
        "created_at": now,
        "updated_at": now,
        "lead_time": 23.178,  # placeholder
        "prediction": {},
        "result_count": len(results),
        "unique_id": os.urandom(8).hex(),
        "import_id": None,
        "last_action": None,
        "bulk_created": False,
        "task": task_id,
        "project": project_id,
        "updated_by": user_id,
        "parent_prediction": None,
        "parent_annotation": None,
        "last_created_by": None,
    }

    # Task object (matches your required format)
    task = {
        "id": task_id,
        "annotations": [annotation],
        "file_upload": os.path.basename(rgb_path),
        "drafts": [],
        "predictions": [],
        "data": {
            "image": f"/data/upload/{project_id}/{os.path.basename(rgb_path)}"
        },
        "meta": {},
        "created_at": now,
        "updated_at": now,
        "inner_id": task_id,
        "total_annotations": 1,
        "cancelled_annotations": 0,
        "total_predictions": 0,
        "comment_count": 0,
        "unresolved_comment_count": 0,
        "last_comment_updated_at": None,
        "project": project_id,
        "updated_by": user_id,
        "comment_authors": []
    }

    with open(output_json, "w") as f:
        json.dump([task], f, indent=2)

    print(f"✅ Saved full Label Studio JSON with {len(results)} masks → {output_json}")


# Example usage
rgb_mask_to_full_json(
    rgb_path=r"C:\Users\Admin\Downloads\Masks\1A_T_1_rangeDataFiltered-0000025-_crack.png",
    output_json=r"C:\Users\Admin\Downloads\output_labelstudio.json"
)
