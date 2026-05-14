import os
import cv2
import sys
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))

from models.unet.UnetPP import UNetPP


# -------------------------------
# CLASS COLOR MAP
# -------------------------------
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


# -------------------------------
# COLORIZE MASK
# -------------------------------
def colorize_prediction(prediction):
    color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    for class_id, color in CLASS_COLOR_MAP.items():
        color_mask[prediction == class_id] = color

    return color_mask


# -------------------------------
# OVERLAY MASK ON IMAGE
# -------------------------------
def overlay_mask_on_image(image, color_mask, alpha=0.5):

    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay


# -------------------------------
# REMOVE SMALL COMPONENTS
# -------------------------------
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


# -------------------------------
# TILE IMAGE
# -------------------------------
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

            # adjust border tiles
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


# -------------------------------
# STITCH TILES
# -------------------------------
def stitch_tiles(pred_tiles, coords, full_size):

    H, W = full_size

    stitched = np.zeros((H, W), dtype=np.uint8)

    count_map = np.zeros((H, W), dtype=np.uint8)

    for pred, (y1, y2, x1, x2) in zip(pred_tiles, coords):

        stitched[y1:y2, x1:x2] += pred

        count_map[y1:y2, x1:x2] += 1

    count_map[count_map == 0] = 1

    stitched = (stitched / count_map).astype(np.uint8)

    return stitched


# -------------------------------
# PREDICT TILE
# -------------------------------
def predict_tile(tile, model, device, mean, std):

    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    transformed = transform(image=tile)

    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        if isinstance(output, dict) and "out" in output:
            pred = output["out"]
        else:
            pred = output

        pred_class = pred.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred_class = cv2.resize(
            pred_class,
            (tile.shape[1], tile.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    pred_class = remove_small_components_multiclass(
        pred_class,
        min_area=10000
    )

    return pred_class


# -------------------------------
# PROCESS SINGLE IMAGE
# -------------------------------
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

    img = cv2.imread(img_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    base_name = os.path.splitext(os.path.basename(img_path))[0]


    # SAVE ORIGINAL IMAGE
    original_save_path = os.path.join(
        save_dir,
        f"{base_name}_original.jpg"
    )

    cv2.imwrite(original_save_path, img)


    # TILE IMAGE
    tiles, coords = tile_image(
        img_rgb,
        tile_size,
        overlap
    )


    pred_tiles = []


    for idx, tile in enumerate(tiles):

        pred_tile = predict_tile(
            tile,
            model,
            device,
            mean,
            std
        )

        pred_tiles.append(pred_tile)
    # STITCH RESULT
    stitched_mask = stitch_tiles(
        pred_tiles,
        coords,
        full_size=img_rgb.shape[:2]
    )
    stitched_color = colorize_prediction(stitched_mask)
    stitched_overlay = overlay_mask_on_image(
        img_rgb,
        stitched_color,
        alpha=0.5
    )


    # SAVE OUTPUTS
    mask_save_path = os.path.join( save_dir, f"{base_name}.png")
    overlay_save_path = os.path.join( save_dir, f"{base_name}_overlay.png")
    cv2.imwrite(mask_save_path, cv2.cvtColor(stitched_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_save_path, cv2.cvtColor(stitched_overlay, cv2.COLOR_RGB2BGR) )


# -------------------------------
# PROCESS FOLDER
# -------------------------------
def process_folder(img_folder,save_dir,model,device,mean,std,tile_size=1024,overlap=256):
    os.makedirs(save_dir, exist_ok=True)
    img_files = [ f for f in os.listdir(img_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for img_file in img_files:
        print(f"Processing {img_file} ..."
        process_image(
            os.path.join(img_folder, img_file),save_dir,model,device,mean,std,tile_size, overlap))
        print(f"Done {img_file}")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda:0"if torch.cuda.is_available()else "cpu")
    weights_path = r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_MIX_384\UNET384_best_epoch6_dice0.860.pth"
    model = UNetPP(
        in_channels=3,
        num_classes=15 )
    weights = torch.load(weights_path, map_location=device)
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()
    mean = (0.478, 0.478, 0.478)
    std = (0.145, 0.145, 0.145)
    img_folder = r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_HIGH_RES"
    save_dir = r"C:\Users\Admin\Downloads\Newfolder"
    process_folder(
        img_folder,
        save_dir,
        model,
        device,
        mean,
        std,
        tile_size=1024,
        overlap=0
    )