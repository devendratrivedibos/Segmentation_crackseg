import json
import os
import pdb

import numpy as np
from PIL import Image, ImageDraw
from label_studio_converter.brush import decode_rle

COLOR_MAP = {

    "alligator crack": (255, 0, 0),  # Red
    "transverse crack": (0, 0, 255),  # Blue
    "transverse": (0, 0, 255),  # Blue
    "transevrse crack": (0, 0, 255),  # Blue
    "transeverse crack": (0, 0, 255),  # Blue
    "longitudinal crack": (0, 255, 0),  # Green
    "longitudnal crack": (0, 255, 0),  # Green
    "longitudunal crack": (0, 255, 0),  # Green
    "longitadinal crack": (0, 255, 0),  # Green
    "longitudinal": (0, 255, 0),  # Green
    "multiple crack": (255, 0, 255),  # Yellow
    "multiple": (255, 0, 255),  # Yellow
    "pothole": (139, 69, 19),  # Brown
    "pathhole": (139, 69, 19),  # Brown
    "patch": (255, 165, 0),  # Orange
    "punchout": (128, 0, 128),  # Purple
    "punchout crack": (128, 0, 128),  # Purple
    "spalling": (0, 255, 255),  # Cyan
    "spallling": (0, 255, 255),  # Cyan
    "corner break": (0, 128, 0),  # Dark green
    "corner crack": (0, 128, 0),  # Dark green
    "croner crack": (0, 128, 0),  # Dark green
    "corner breack": (0, 128, 0),  # Dark green
    "joint sealed transverse": (255, 100, 203),  # Light pink
    "joint sealed transeverse": (255, 100, 203),  # Light pink
    "joint sealed transverse crack": (255, 100, 203),  # Light pink
    "joint seal transverse": (255, 100, 203),  # Light pink
    "jiont sealed transverse": (255, 100, 203),  # Light pink
    "joint sealed transvers": (255, 100, 203),  # Light pink
    "join sealed transvrse": (255, 100, 203),  # Light pink
    "join seal transevrse": (255, 100, 203),  # Light pink
    "join sealed transvers": (255, 100, 203),  # Light pink
    "joint sealed longitudinal": (199, 21, 133),  # Dark pink
    "joint sealed longitudinal crack": (199, 21, 133),  # Dark pink
    "joint sealed longtudinal": (199, 21, 133),  # Dark pink
    "joint sealed longitudnal": (199, 21, 133),  # Dark pink
    "joint seal longitudinal": (199, 21, 133),  # Dark pink
    "joint seal longitudnal": (199, 21, 133),  # Dark pink
    "jiont sealed longitadinal": (199, 21, 133),  # Dark pink
    "cracking": (255, 215, 0),  # Gold
    "unclassified": (255, 255, 255),  # White
    "popsout": (112, 102, 255),
    "popout": (112, 102, 255),
    "popout crack": (112, 102, 255),
}

# track label
unique_labels = set()


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
                # track label
                unique_labels.add(label)

                color = COLOR_MAP.get(label.lower(), (255, 255, 255))

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

        parts = base_name.split("-", 1)  # split only on the first dash
        if len(parts) == 2:
            out_name = parts[1]
        # Save mask for this image
        out_name = f"{out_name}.png"
        out_path = os.path.join(output_dir, out_name)
        out_img = Image.fromarray(mask_img.astype(np.uint8))
        out_img.save(out_path, format="PNG")
        print(f"Saved {out_path}")

    print("\n=== Unique Labels Found ===")
    for lbl in sorted(unique_labels):
        print(lbl)


folderlist = [
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\1\PRASAD_(26-9-25)\project-30-at-2025-09-26-13-40-46bb13ee.json",
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\2\Bushra 26.09.2025\project-48-at-2025-09-26-13-10-d8860596.json",
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\3\GAYATRI 26.09.2025\project-35-at-2025-09-26-12-12-c226331c.json",
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\4\RUTUJA (26-09-2025)\project-30-at-2025-09-26-13-19-3f836136.json",
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\5\PRATHAMESH 26-9-2025\project-27-at-2025-09-26-13-08-23e1c7cf (1).json",
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\6\payal s1_p6_26.09.2025\project-31-at-2025-09-26-15-47-07411699.json",
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\7\Jayesh 26-9-2025\project-35-at-2025-09-26-13-53-25045b69.json",
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\8\Akash 26.09.2025\project-35-at-2025-09-26-15-02-9fc22ec4.json",
r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORK_divided\9\Sharan 09-26-2025\project-3-at-2025-09-26-13-48-3b80c742.json",
]
for i in folderlist:
        save_masks_from_json(i,
                             r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\REWORk_MASKS",
                             use_rgb=True)