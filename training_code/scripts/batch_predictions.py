from tqdm import tqdm
import cv2
import numpy as np
import torch
from PIL import Image
import sys
import os
from torchvision import transforms as T
import train_utils.transforms as T
from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
from models.segformer.segformer import SegFormer
from models.unet.unet import UNet
from models.unet.mobilenet_unet import MobileV3Unet
from models.unet.vgg_unet import VGG16UNet
from models.fcn.fcn import fcn_resnet101
from models.unet.UnetPP import UNetPP

CLASS_COLOR_MAP = {
    0: [0, 0, 0],  # Black   - Background
    # 1:  [255, 0, 0],      # Red     - Alligator
    # 2:  [0, 0, 255],      # Blue    - Transverse Crack
    # 3:  [0, 255, 0],      # Green   - Longitudinal Crack
    # 4:  [139, 69, 19],    # Brown   - Pothole
    # 5:  [255, 165, 0],    # Orange  - Patches
    # 6:  [128, 0, 128],    # Purple  - Punchout
    # 7:  [0, 255, 255],    # Cyan    - Spalling
    # 8:  [0, 128, 0],      # Dark Green - Corner Break
    9: [255, 100, 203],  # Light Pink - Sealed Joint - T
    10: [199, 21, 133],  # Dark Pink  - Sealed Joint - L
    # 11: [255, 215, 0],    # Gold       - Cracking
    # 12: [255, 255, 255],  # White      - Unclassified
    # 13: [255, 0, 255],    # Yellow     - multiple crack
}


def main(imgs_root=None, prediction_save_path=None, batch_size=4):
    num_classes = 13 + 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = (0.493, 0.493, 0.493)
    std = (0.144, 0.144, 0.144)
    data_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # Collect images
    images_list = [img for img in os.listdir(imgs_root) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    os.makedirs(prediction_save_path, exist_ok=True)

    # Model
    model = UNetPP(in_channels=3, num_classes=num_classes)
    weights_path = r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_concrete_13\UNET_concrete_13_best_epoch171_dice0.902.pth"
    pretrain_weights = torch.load(weights_path, map_location=device)
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)
    model.eval()

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(images_list), batch_size), desc=f"Processing {os.path.basename(imgs_root)}"):
            batch_files = images_list[i:i + batch_size]
            batch_imgs, orig_names = [], []

            for image in batch_files:
                original_img = cv2.imread(os.path.join(imgs_root, image))
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                img, _ = data_transform(original_img, original_img)
                batch_imgs.append(img)
                orig_names.append(image)

            # stack and forward pass
            batch_tensor = torch.stack(batch_imgs).to(device)
            outputs = model(batch_tensor)
            preds = outputs['out'].argmax(1).cpu().numpy().astype(np.uint8)

            # postprocess + save
            for pred, fname in zip(preds, orig_names):
                pred = remove_small_components_multiclass(pred, min_area=400)
                pred_color = colorize_prediction(pred)
                save_path = os.path.join(prediction_save_path, fname.split('.')[0] + '.png')
                cv2.imwrite(save_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

    print("✅ Processing done for", imgs_root)


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
    main(imgs_root=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-6\process_distress",
         prediction_save_path=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-6\SECTION_RESULTS",
         batch_size=8)

    main(imgs_root=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-7\process_distress",
         prediction_save_path=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-7\SECTION_RESULTS",
         batch_size=8)

    main(imgs_root=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-8\process_distress",
         prediction_save_path=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-8\SECTION_RESULTS",
         batch_size=8)

    main(imgs_root=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-9\process_distress",
         prediction_save_path=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-9\SECTION_RESULTS",
         batch_size=8)

    main(imgs_root=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-10\process_distress",
         prediction_save_path=r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\SECTION-10\SECTION_RESULTS",
         batch_size=8)
