import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.unet.UnetPP import UNetPP

CLASS_COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 0, 255],
    3: [0, 255, 0],
    4: [139, 69, 19],
    5: [255, 165, 0],
    12: [112, 102, 255]
}

def colorize_prediction(prediction):
    color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLOR_MAP.items():
        color_mask[prediction == class_id] = color
    return color_mask

def overlay_mask_on_image(image, color_mask, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

def remove_small_components_multiclass(mask, min_area=400):
    cleaned = np.zeros_like(mask, dtype=mask.dtype)
    for cls in np.unique(mask):
        if cls == 0:
            continue
        class_mask = (mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = cls
    return cleaned

def tile_image(img, tile_size=1024, overlap=256):
    H, W = img.shape[:2]
    step = tile_size - overlap
    tiles, coords = [], []
    for y in range(0, H, step):
        for x in range(0, W, step):
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size); y2 = y1 + tile_size
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size); x2 = x1 + tile_size
            tile = img[y1:y2, x1:x2].copy()
            tiles.append(tile)
            coords.append((y1, y2, x1, x2))
    return tiles, coords

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

def predict_tile(tile, model, device, mean, std):
    resize_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    transformed = resize_transform(image=tile)
    input_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict) and "out" in output:
            pred = output["out"]
        else:
            pred = output
        pred_class = pred.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_class = cv2.resize(pred_class, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_NEAREST)
    return remove_small_components_multiclass(pred_class, min_area=10000)

def process_image(img_path, save_dir, model, device, mean, std, tile_size=1024, overlap=256):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tiles, coords = tile_image(img_rgb, tile_size, overlap)
    pred_tiles = []

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for idx, tile in enumerate(tiles):
        pred_tile = predict_tile(tile, model, device, mean, std)
        pred_tiles.append(pred_tile)

        # Save tile, prediction, overlay in same folder
        # tile_save = os.path.join(save_dir, f"{base_name}_tile{idx:03d}.jpg")
        # cv2.imwrite(tile_save, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

        # pred_color = colorize_prediction(pred_tile)
        # pred_save = os.path.join(save_dir, f"{base_name}_tile{idx:03d}_pred.png")
        # cv2.imwrite(pred_save, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

        # overlay = overlay_mask_on_image(tile, pred_color, alpha=0.5)
        # overlay_save = os.path.join(save_dir, f"{base_name}_tile{idx:03d}_overlay.png")
        # cv2.imwrite(overlay_save, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    stitched = stitch_tiles(pred_tiles, coords, full_size=img_rgb.shape[:2])
    stitched_color = colorize_prediction(stitched)
    stitched_overlay = overlay_mask_on_image(img_rgb, stitched_color, alpha=0.5)

    cv2.imwrite(os.path.join(save_dir, f"{base_name}.png"), cv2.cvtColor(stitched_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_overlay.png"), cv2.cvtColor(stitched_overlay, cv2.COLOR_RGB2BGR))

def process_folder(img_folder, save_dir, model, device, mean, std, tile_size=1024, overlap=256):
    os.makedirs(save_dir, exist_ok=True)
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for img_file in img_files:
        print(f"Processing {img_file}...")
        process_image(os.path.join(img_folder, img_file), save_dir, model, device, mean, std, tile_size, overlap)
        print(f"✅ Done: {img_file}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_384\UNET384_best_epoch4_dice0.926.pth"

    model = UNetPP(in_channels=3, num_classes=4)
    w = torch.load(weights_path, map_location=device)
    if "model" in w:
        model.load_state_dict(w["model"])
    else:
        model.load_state_dict(w)
    model.to(device).eval()

    mean = (0.456, 0.456, 0.456)
    std = (0.145, 0.145, 0.145)

    img_folder = r"W:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\process_distress_HIGH_RES"
    save_dir = r"C:\Users\Admin\Downloads\Newfolder"
    process_folder(img_folder, save_dir, model, device, mean, std, tile_size=1024, overlap=256)
