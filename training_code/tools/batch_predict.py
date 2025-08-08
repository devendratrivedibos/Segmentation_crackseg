import os
import pdb
import time
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))
import cv2
import numpy as np
import torch
from PIL import Image
# from torchvision import transforms as T
import train_utils.transforms as T
import sys


from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
from models.segformer.segformer import SegFormer
from models.unet.unet import UNet
from models.unet.mobilenet_unet import MobileV3Unet
from models.unet.vgg_unet import VGG16UNet
from models.fcn.fcn import fcn_resnet101
from models.unet.UnetPP import UNetPP

# Define color map for classes (BGR for OpenCV)
CLASS_COLOR_MAP = {
    0: [0, 0, 0],        # background: black
    1: [255, 0, 0],      # red (BGR) for class 1
    2: [255, 255, 255],
    3: [0, 255, 0],  #
    4: [0, 0, 255],  #
}


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1 + 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = T.Compose([
        # T.Resize(512),
         T.ToTensor(),
         T.Normalize([0.473, 0.493, 0.504], [0.100, 0.100, 0.099])])

    # load image

    imgs_root = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_SPLIT_OLD\VAL\IMAGES"
    images_list = os.listdir(imgs_root)
    prediction_save_path = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\UNETPP"
    os.makedirs(prediction_save_path, exist_ok=True)

    # create model
    model = UNetPP(in_channels=3, num_classes=num_classes)
    # model = VGG16UNet(num_classes=num_classes)
    # model = MobileV3Unet(num_classes=num_classes)

    # load model weights
    weights_path = r"D:\Devendra_Files\CrackSegFormer-main\weights\UNetPP_1\UnetPP_1_best_epoch77_dice0.568.pth"
    # weights_path = "segformer/20250805-194612-best_model.pth"

    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."

    pretrain_weights = torch.load(weights_path, map_location='cuda:0')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)

    # model = fcn_resnet101(aux=False, num_classes=num_classes)
    # model = deeplabv3_resnet101(aux=False, num_classes=num_classes)
    # model = deeplabv3_mobilenetv3_large(aux=False, num_classes=num_classes)
    # # deeplabv3:delete weights about aux_classifier
    # # weights_dict = torch.load(args.weights, map_location='cpu')['model']
    # weights_dict = torch.load(weights_path, map_location='cpu')
    # for k in list(weights_dict.keys()):
    #     if "aux" in k:
    #         del weights_dict[k]
    #
    # model.load_state_dict(weights_dict)
    # model.to(device)

    # prediction
    model.eval()
    with torch.no_grad():
        for index, image in enumerate(images_list):
            original_img = cv2.imread(os.path.join(imgs_root, image))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.resize(original_img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            img , _= data_transform(original_img, original_img)
            img = torch.unsqueeze(img, dim=0)

            output = model(img.to(device))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            prediction_color = colorize_prediction(prediction)
            mask_save_path = os.path.join(prediction_save_path, image.split('.')[0] + '_mask.png')
            cv2.imwrite(mask_save_path, cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR))

            # Overlay on original image and save overlay
            original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
            overlaid = overlay_mask_on_image(original_img_bgr, cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR))
            overlay_save_path = os.path.join(prediction_save_path, image.split('.')[0] + '_overlay.png')
            cv2.imwrite(overlay_save_path, overlaid)

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


if __name__ == '__main__':
    main()
