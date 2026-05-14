import os
import cv2
import numpy as np
import torch
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))
from models.unet.UnetPP import UNetPP


# --------------------------------
# CLASS COLOR MAP
# --------------------------------
CLASS_COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 0, 255],
    3: [0, 255, 0],
    4: [139, 69, 19],
    5: [255, 165, 0],
    6: [255, 0, 255],
    7: [0, 255, 255],
    8: [0, 128, 0],
    9: [255, 100, 203],
    10: [199, 21, 133],
    11: [128, 0, 128],
    12: [112, 102, 255],
    13: [255, 255, 255],
    14: [255, 215, 0],
}


# --------------------------------
# UTILS
# --------------------------------
def colorize_prediction(prediction):

    color_mask = np.zeros(
        (prediction.shape[0], prediction.shape[1], 3),
        dtype=np.uint8
    )

    for class_id, color in CLASS_COLOR_MAP.items():

        color_mask[prediction == class_id] = color

    return color_mask


def overlay_mask_on_image(image, color_mask, alpha=0.5):

    return cv2.addWeighted(
        image,
        1 - alpha,
        color_mask,
        alpha,
        0
    )


def remove_small_components_multiclass(mask, min_area=400):

    cleaned = np.zeros_like(mask, dtype=mask.dtype)

    for cls in np.unique(mask):

        if cls == 0:
            continue

        class_mask = (mask == cls).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            class_mask,
            connectivity=8
        )

        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area >= min_area:

                cleaned[labels == i] = cls

    return cleaned


# --------------------------------
# TILE IMAGE
# --------------------------------
def tile_image(img, tile_size=1024, overlap=256):

    H, W = img.shape[:2]

    step = tile_size - overlap

    tiles = []
    coords = []

    for y in range(0, H, step):

        for x in range(0, W, step):

            y1 = y
            x1 = x

            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)

            if y2 - y1 < tile_size:

                y1 = max(0, y2 - tile_size)
                y2 = y1 + tile_size

            if x2 - x1 < tile_size:

                x1 = max(0, x2 - tile_size)
                x2 = x1 + tile_size

            tile = img[y1:y2, x1:x2].copy()

            tiles.append(tile)

            coords.append((y1, y2, x1, x2))

    return tiles, coords


# --------------------------------
# STITCH MASK
# --------------------------------
def stitch_tiles(pred_tiles, coords, full_size):

    H, W = full_size

    stitched = np.zeros((H, W), dtype=np.float32)

    count_map = np.zeros((H, W), dtype=np.float32)

    for pred, (y1, y2, x1, x2) in zip(pred_tiles, coords):

        stitched[y1:y2, x1:x2] += pred

        count_map[y1:y2, x1:x2] += 1

    count_map[count_map == 0] = 1

    stitched = (stitched / count_map).astype(np.uint8)

    return stitched


# --------------------------------
# LOAD MODEL
# --------------------------------
def load_model(weights_path, device):

    model = UNetPP(
        in_channels=3,
        num_classes=15
    )

    weights = torch.load(
        weights_path,
        map_location=device
    )

    if "model" in weights:

        model.load_state_dict(
            weights["model"]
        )

    else:

        model.load_state_dict(
            weights
        )

    model.to(device)

    model.eval()

    return model


# --------------------------------
# BATCH PREDICT TILES
# --------------------------------
def predict_tiles_batch(
        tiles,
        coords,
        model,
        device,
        mean,
        std,
        tile_dir,
        batch_size=8
):

    transform = A.Compose([

        A.Resize(384, 384),

        A.Normalize(
            mean=mean,
            std=std
        ),

        ToTensorV2()

    ])

    preds = []

    for i in range(0, len(tiles), batch_size):

        batch_tiles = tiles[i:i+batch_size]

        batch_coords = coords[i:i+batch_size]

        batch_tensor = []

        for tile in batch_tiles:

            transformed = transform(
                image=tile
            )

            batch_tensor.append(
                transformed["image"]
            )

        batch_tensor = torch.stack(
            batch_tensor
        ).to(device)

        with torch.no_grad():

            output = model(
                batch_tensor
            )

            if isinstance(output, dict) and "out" in output:

                output = output["out"]

            pred_batch = output.argmax(1).cpu().numpy()

        for j, (pred, tile, coord) in enumerate(
                zip(pred_batch, batch_tiles, batch_coords)
        ):

            tile_index = i + j

            pred = cv2.resize(

                pred.astype(np.uint8),

                (tile.shape[1], tile.shape[0]),

                interpolation=cv2.INTER_NEAREST

            )

            pred = remove_small_components_multiclass(

                pred,
                min_area=400

            )

            preds.append(pred)

            # -------------------------
            # SAVE TILE RESULTS
            # -------------------------

            color_mask = colorize_prediction(pred)

            overlay = overlay_mask_on_image(

                tile,

                color_mask,

                alpha=0.5

            )

            cv2.imwrite(

                os.path.join(

                    tile_dir,

                    f"tile_{tile_index:04d}_original.jpg"

                ),

                cv2.cvtColor(

                    tile,

                    cv2.COLOR_RGB2BGR

                )

            )

            cv2.imwrite(

                os.path.join(

                    tile_dir,

                    f"tile_{tile_index:04d}.png"

                ),

                cv2.cvtColor(

                    color_mask,

                    cv2.COLOR_RGB2BGR

                )

            )

            cv2.imwrite(

                os.path.join(

                    tile_dir,

                    f"tile_{tile_index:04d}_overlay.png"

                ),

                cv2.cvtColor(

                    overlay,

                    cv2.COLOR_RGB2BGR

                )

            )

    return preds


# --------------------------------
# PROCESS SINGLE IMAGE
# --------------------------------
def process_image(

        img_path,

        save_dir,

        model,

        device,

        mean,

        std,

        tile_size=1024,

        overlap=256

):

    img = cv2.imread(

        img_path

    )

    img_rgb = cv2.cvtColor(

        img,

        cv2.COLOR_BGR2RGB

    )

    base_name = os.path.splitext(

        os.path.basename(img_path)

    )[0]

    tile_dir = os.path.join(

        save_dir,

        "tiles",

        base_name

    )

    os.makedirs(

        tile_dir,

        exist_ok=True

    )

    tiles, coords = tile_image(

        img_rgb,

        tile_size,

        overlap

    )

    pred_tiles = predict_tiles_batch(

        tiles,

        coords,

        model,

        device,

        mean,

        std,

        tile_dir,

        batch_size=8

    )

    stitched_mask = stitch_tiles(

        pred_tiles,

        coords,

        full_size=img_rgb.shape[:2]

    )

    stitched_color = colorize_prediction(

        stitched_mask

    )

    overlay = overlay_mask_on_image(

        img_rgb,

        stitched_color,

        alpha=0.5

    )

    cv2.imwrite(

        os.path.join(

            save_dir,

            f"{base_name}.png"

        ),

        cv2.cvtColor(

            stitched_color,

            cv2.COLOR_RGB2BGR

        )

    )

    cv2.imwrite(

        os.path.join(

            save_dir,

            f"{base_name}_overlay.png"

        ),

        cv2.cvtColor(

            overlay,

            cv2.COLOR_RGB2BGR

        )

    )

    return base_name


# --------------------------------
# PROCESS FOLDER
# --------------------------------
def process_folder(

        img_folder,

        save_dir,

        weights_path,

        device,

        mean,

        std,

        tile_size=1024,

        overlap=256

):

    os.makedirs(

        save_dir,

        exist_ok=True

    )

    model = load_model(

        weights_path,

        device

    )

    img_files = [

        f for f in os.listdir(

            img_folder

        )

        if f.lower().endswith(

            (".jpg", ".jpeg", ".png")

        )

    ]

    for img_file in tqdm(

            img_files,

            desc="Processing Images"

    ):

        process_image(

            os.path.join(

                img_folder,

                img_file

            ),

            save_dir,

            model,

            device,

            mean,

            std,

            tile_size,

            overlap

        )

    print("Done")


# --------------------------------
# MAIN
# --------------------------------
if __name__ == "__main__":

    device = torch.device(

        "cuda:0"

        if torch.cuda.is_available()

        else "cpu"

    )

    weights_path = r"D:\Devendra_Files\segmentation_training\weights\UNET_concrete_tile\UNET_concrete_tile_best_epoch73_dice0.927.pth"

    mean = (0.488, 0.488, 0.488)

    std = (0.145, 0.145, 0.145)

    img_folder = r"Z:\Devendra\CONCRETE\HIGH_RES_IMAGES"

    save_dir = r"Z:\Devendra\CONCRETE\HIGH_RES_IMAGES\OUTPUT"

    process_folder(

        img_folder,

        save_dir,

        weights_path,

        device,

        mean,

        std,

        tile_size=1024,

        overlap=0

    )