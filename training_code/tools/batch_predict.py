import os
import time
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))
import cv2
import numpy as np
import torch
import train_utils.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.unet.UnetPP import UNetPP

CLASS_COLOR_MAP = {
    0: [0, 0, 0],  # Black   - Background
    1: [255, 0, 0],  # Red     - Alligator
    2:  [0, 0, 255],      # Blue    - Transverse Crack
    3:  [0, 255, 0],      # Green   - Longitudinal Crack
    4:  [139, 69, 19],    # Brown   - Pothole
    5:  [255, 165, 0],    # Orange  - Patches
    # 4: [255, 0, 255],  # violet     - multiple crack
    # 5: [255, 0, 255],  # Grey       - popout
    # 7:  [0, 255, 255],    # Cyan    - Spalling
    # 8:  [0, 128, 0],      # Dark Green - Corner Break
    # 9: [255, 100, 203],  # Light Pink - Sealed Joint - T
    # 10: [199, 21, 133],  # Dark Pink  - Sealed Joint - L
    # 11:  [128, 0, 128],    # Purple  - Punchout
    12: [112, 102, 255],  # popout Grey
    # 13: [255, 255, 255],  # White      - Unclassified
    # 14: [255, 215, 0],    # Gold       - Cracking
}


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(imgs_root=None, prediction_save_path=None, weights_path=None):
    num_classes = 3 + 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = (0.456, 0.456, 0.456)
    std = (0.145, 0.145, 0.145)

    data_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(), ])

    imgs_root = imgs_root
    images_list = os.listdir(imgs_root)
    images_list = [img for img in images_list if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    weights_path = weights_path
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    pretrain_weights = torch.load(weights_path, map_location='cuda:0')

    prediction_save_path = prediction_save_path
    os.makedirs(prediction_save_path, exist_ok=True)

    # create model
    model = UNetPP(in_channels=3, num_classes=num_classes)

    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for index, image in enumerate(images_list):
            original_img = cv2.imread(os.path.join(imgs_root, image))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            transformed = data_transform(image=original_img)
            img = transformed["image"]
            img = torch.unsqueeze(img, dim=0)
            output = model(img.to(device))
            # handle dict or tensor
            if isinstance(output, dict) and "out" in output:
                pred = output["out"]
            elif isinstance(output, torch.Tensor):
                pred = output
            else:
                raise RuntimeError("Unexpected model output format")
            prediction = pred.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            prediction = remove_small_components_multiclass(prediction, min_area=400)
            prediction_color = colorize_prediction(prediction)
            mask_save_path = os.path.join(prediction_save_path, image.split('.')[0] + '.png')
            cv2.imwrite(mask_save_path, cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR))

            # Overlay on original image and save overlay
            # original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
            # overlaid = overlay_mask_on_image(original_img_bgr, cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR))
            # overlay_save_path = os.path.join(prediction_save_path, image.split('.')[0] + '_overlay.png')
            # cv2.imwrite(overlay_save_path, overlaid)

            print(f"\r[{image}] processed overlay [{index + 1}/{len(images_list)}]", end="")

        print("\nprocessing done!")
        print()

    print("processing done!")


def colorize_prediction(prediction):
    """
    Convert class-wise prediction (H, W) to RGB image.
    """
    color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLOR_MAP.items():
        color_mask[prediction == class_id] = color
    return color_mask


def overlay_mask_on_image(image, color_mask, alpha=0.5):
    """
    Overlay a multi-class RGB mask on a color image.
    """
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay


def remove_small_components_multiclass(mask, min_area=400):
    """
    Removes small connected components per class in a multi-class segmentation mask.

    Args:
        mask (np.ndarray): Segmentation mask (H×W, dtype int).
                           0 = background, 1..N = classes.
        min_area (int): Minimum number of pixels to keep.
        min_width (int): Minimum bounding box width.
        min_height (int): Minimum bounding box height.

    Returns:
        np.ndarray: Cleaned segmentation mask.
    """
    cleaned = np.zeros_like(mask, dtype=mask.dtype)

    for cls in np.unique(mask):
        if cls == 0:  # skip background
            continue

        class_mask = (mask == cls).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = cls

    return cleaned


if __name__ == '__main__':
    main(imgs_root=r"C:\Users\Admin\Downloads",
         prediction_save_path=r"C:\Users\Admin\Downloads\Newfolder",
         weights_path= r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_384\UNET384_best_epoch4_dice0.926.pth")
