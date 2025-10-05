import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.unet.UnetPP import UNetPP


# ---- CLASS COLOR MAP ----
CLASS_COLOR_MAP = {
    0: [0, 0, 0],      # Background
    1: [255, 0, 0],    # Alligator
    2: [0, 0, 255],    # Transverse Crack
    3: [0, 255, 0],    # Longitudinal Crack
    4: [139, 69, 19],  # Pothole
    5: [255, 165, 0],  # Patches
    12: [112, 102, 255],  # Popout
}


# ---- TILING ----
def tile_image(image, tile_size=1024, overlap=0):
    h, w, _ = image.shape
    tiles, coords = [], []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = image[y:y2, x:x2]
            pad = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tiles.append(pad)
            coords.append((x, y, tile.shape[1], tile.shape[0]))
    return tiles, coords, (h, w)


def stitch_tiles(tiles, coords, full_size, num_classes):
    H, W = full_size
    prob_map = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((num_classes, H, W), dtype=np.float32)

    for probs, (x, y, w, h) in zip(tiles, coords):
        probs = probs[:, :h, :w]  # remove padding
        prob_map[:, y:y+h, x:x+w] += probs
        count_map[:, y:y+h, x:x+w] += 1

    prob_map /= np.maximum(count_map, 1e-6)
    return np.argmax(prob_map, axis=0).astype(np.uint8)


def colorize_prediction(prediction):
    color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLOR_MAP.items():
        color_mask[prediction == class_id] = color
    return color_mask


def remove_small_components_multiclass(mask, min_area=400):
    cleaned = np.zeros_like(mask, dtype=mask.dtype)
    for cls in np.unique(mask):
        if cls == 0:
            continue
        class_mask = (mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = cls
    return cleaned


def overlay_mask_on_image(image, color_mask, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


# ---- MAIN ----
def main(imgs_root=None, prediction_save_path=None, weights_path=None):
    num_classes = 5 + 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean, std = (0.456, 0.456, 0.456), (0.145, 0.145, 0.145)
    # mean, std = (0.488, 0.488, 0.488), (0.149, 0.149, 0.149)

    data_transform = A.Compose([
        # A.Resize(384, 384),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Load model
    model = UNetPP(in_channels=3, num_classes=num_classes)
    pretrain_weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(pretrain_weights["model"] if "model" in pretrain_weights else pretrain_weights)
    model.to(device).eval()

    os.makedirs(prediction_save_path, exist_ok=True)
    images_list = [f for f in os.listdir(imgs_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with torch.no_grad():
        for index, image_name in enumerate(images_list):
            original_img = cv2.cvtColor(cv2.imread(os.path.join(imgs_root, image_name)), cv2.COLOR_BGR2RGB)

            # ---- TILE ----
            tiles, coords, full_size = tile_image(original_img, tile_size=1024, overlap=128)

            pred_tiles = []
            for t_index, tile in enumerate(tiles):
                transformed = data_transform(image=tile)
                img_tensor = transformed["image"].unsqueeze(0).to(device)

                pred = model(img_tensor)
                if isinstance(pred, dict) and "out" in pred:
                    pred = pred["out"]

                probs = torch.softmax(pred, dim=1).squeeze(0).cpu().numpy()

                # resize prediction back to 1024
                probs_resized = np.zeros((num_classes, 1024, 1024), dtype=np.float32)
                for c in range(num_classes):
                    probs_resized[c] = cv2.resize(probs[c], (1024, 1024), interpolation=cv2.INTER_LINEAR)

                # ---- per tile predicted class mask ----
                tile_mask = np.argmax(probs_resized, axis=0).astype(np.uint8)
                tile_mask_color = colorize_prediction(tile_mask)

                # ---- save per-tile outputs ----
                tile_base = os.path.splitext(image_name)[0] + f"_tile{t_index}"
               # cv2.imwrite(os.path.join(prediction_save_path, tile_base + "_mask.png"), tile_mask)
                #cv2.imwrite(os.path.join(prediction_save_path, tile_base + "_mask_rgb.png"),cv2.cvtColor(tile_mask_color, cv2.COLOR_RGB2BGR))
                overlay = overlay_mask_on_image(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                                                cv2.cvtColor(tile_mask_color, cv2.COLOR_RGB2BGR))
               # cv2.imwrite(os.path.join(prediction_save_path, tile_base + "_overlay.png"), overlay)

                pred_tiles.append(probs_resized)

            # ---- STITCH ----
            stitched_pred = stitch_tiles(pred_tiles, coords, full_size, num_classes)

            # ---- POSTPROCESS ----
            cleaned = remove_small_components_multiclass(stitched_pred, min_area=400)
            prediction_color = colorize_prediction(cleaned)

            # ---- SAVE FINAL MASK & OVERLAY ----
            mask_save_path = os.path.join(prediction_save_path, image_name.split('.')[0] + '_final_mask.png')
            cv2.imwrite(mask_save_path, cleaned)
            cv2.imwrite(os.path.join(prediction_save_path, image_name.split('.')[0] + '_final_mask_rgb.png'),
                        cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR))
            overlay_final = overlay_mask_on_image(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR),
                                                  cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(prediction_save_path, image_name.split('.')[0] + '_final_overlay.png'),
                        overlay_final)

            print(f"\r[{image_name}] processed [{index + 1}/{len(images_list)}]", end="")

    print("\nProcessing done!")


if __name__ == '__main__':
    main(
        imgs_root=r"C:\Users\Admin\Downloads",
        prediction_save_path=r"C:\Users\Admin\Downloads\Newfolder",
        weights_path=r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_asphalt_1024\UNET_asp_1024_best_epoch243_dice0.705.pth"
        # weights_path=r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_subset\epoch_9_iter_latest.pth"
    )
