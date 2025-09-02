import json
import os
import pdb

import numpy as np
from PIL import Image, ImageDraw
from label_studio_converter.brush import decode_rle

# Define color map per class
LABEL_MAP = {
    "Alligator": (255, 0, 0),   # Red
    "Longitudinal": (0, 255, 0),        # Green
    "Tree": (0, 0, 255),       # Blue
    "unknown": (255, 255, 255) # White fallback
}

def save_masks_from_json(json_file, output_dir="./masks", use_rgb=True):
    with open(json_file, "r") as f:
        tasks = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for task in tasks:
        # Get image name from "data"
        image_path = task["data"].get("image")
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Collect image size from first annotation
        height, width = None, None
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                height = int(result.get("original_height", 0))
                width = int(result.get("original_width", 0))
                if height and width:
                    break
            if height and width:
                break

        if not height or not width:
            print(f"Skipping {base_name}, no dimensions found")
            continue

        # Initialize blank mask
        if use_rgb:
            mask_img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            mask_img = np.zeros((height, width), dtype=np.uint8)

        # Process all annotations for this image
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                value = result.get("value", {})
                label = value.get("brushlabels") or value.get("polygonlabels") or ["unknown"]
                label = label[0]
                color = LABEL_MAP.get(label, (255, 255, 255))

                # --- Case 1: Brush (RLE) ---
                if value.get("format") == "rle":
                    rle_data = value["rle"]

                    # Label Studio uses (W, H) order
                    mask = decode_rle(rle_data, (width, height))
                    mask = np.array(mask)

                    # Try reshaping safely
                    if mask.size == width * height:
                        mask = mask.reshape((height, width))
                    elif mask.size == width * height * 4:  # RGBA case
                        mask = mask.reshape((height, width, 4))
                        mask = mask[:, :, 3]  # use alpha channel
                    else:
                        raise RuntimeError(
                            f"Unexpected RLE mask size {mask.size}, "
                            f"expected {width * height} or {width * height * 4}"
                        )

                    mask = (mask > 0).astype(np.uint8)

                    if use_rgb:
                        mask_img[mask == 1] = color
                    else:
                        mask_img[mask == 1] = 255

                # --- Case 2: Polygon ---
                elif "points" in value:
                    points = value["points"]
                    # Convert percentages to pixels
                    polygon = [(int(x * width / 100), int(y * height / 100)) for x, y in points]

                    pil_mask = Image.new("L", (width, height), 0)
                    ImageDraw.Draw(pil_mask).polygon(polygon, outline=1, fill=1)
                    polygon_mask = np.array(pil_mask)

                    if use_rgb:
                        mask_img[polygon_mask == 1] = color
                    else:
                        mask_img[polygon_mask == 1] = 255

        # Save mask for this image
        out_name = f"{base_name}_mask.png"
        out_path = os.path.join(output_dir, out_name)
        out_img = Image.fromarray(mask_img.astype(np.uint8))
        out_img.save(out_path, format="PNG")
        print(f"Saved {out_path}")


# Run
save_masks_from_json(
    r"C:\Users\Admin\Downloads\multiple_output_labelstudio.json",
    output_dir="./masks",
    use_rgb=True
)
