import json
import os
import numpy as np
from PIL import Image
from label_studio_converter.brush import decode_rle

# Optional: RGB colormap
COLOR_MAP = {
    "alligator crack": (255, 0, 0),  # Red
    "transverse crack": (0, 0, 255),  # Blue
    "longitudinal crack": (0, 255, 0),  # Green
    "multiple crack": (255, 255, 0),  # Yellow
    "pothole": (139, 69, 19),  # Brown
    "patch": (255, 165, 0),  # Orange
    "punchout": (128, 0, 128),  # Purple
    "spalling": (0, 255, 255),  # Cyan
    "corner break": (0, 128, 0),  # Dark green
    "corner crack": (0, 128, 0),  # Dark green
    "joint sealed  transverse": (255, 100, 203),  # Light pink
    "joint sealed transverse": (255, 100, 203),  # Light pink
    "joint sealed longitudinal": (199, 21, 133),  # Dark pink
    "cracking": (255, 215, 0),  # Gold
    "unclassified": (255, 255, 255),  # White
}

def save_brush_masks(json_file, output_dir="./masks", use_rgb=False):
    with open(json_file, "r") as f:
        tasks = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for task in tasks:
        for ann_idx, ann in enumerate(task.get("annotations", [])):
            for res_idx, result in enumerate(ann.get("result", [])):
                value = result.get("value", {})
                if value.get("format") != "rle":
                    continue

                rle_data = value["rle"]
                width = int(result.get("original_width"))
                height = int(result.get("original_height"))
                label = value.get("brushlabels", ["unknown"])[0]

                # Decode brush mask
                mask = decode_rle(rle_data, (height, width))
                mask = np.array(mask)

                # Handle flat or RGBA
                if mask.ndim == 1:
                    if mask.size == height * width * 4:
                        mask = mask.reshape((height, width, 4))
                        mask = mask[:, :, 3]
                    elif mask.size == height * width:
                        mask = mask.reshape((height, width))
                    else:
                        raise RuntimeError(
                            f"Unexpected mask size {mask.size}, "
                            f"expected {height*width} or {height*width*4}"
                        )
                elif mask.ndim == 3 and mask.shape[-1] == 4:
                    mask = mask[:, :, 3]

                # Binarize
                mask = (mask > 0).astype(np.uint8)

                # Save as RGB or grayscale
                if use_rgb:
                    label = label.lower()
                    color = COLOR_MAP.get(label, (255, 255, 255))
                    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_rgb[mask == 1] = color
                    out_img = Image.fromarray(mask_rgb, mode="RGB")
                else:
                    out_img = Image.fromarray(mask * 255, mode="L")

                out_name = f"task{task['id']}_ann{ann_idx}_res{res_idx}.png"
                out_path = os.path.join(output_dir, out_name)
                out_img.save(out_path, format="PNG")
                print(f"Saved {out_path}")

save_brush_masks(r"C:\Users\Admin\Downloads\multiple_output_labelstudio.json", output_dir="./masks", use_rgb=True)