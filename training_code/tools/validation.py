import os
import torch
from torch.utils import data
import time
import datetime
import numpy as np
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))


import torch.nn.functional as F
from train_utils.dice_coefficient_loss import multiclass_dice_coeff, build_target
import train_utils.distributed_utils as utils
from train_utils.my_dataset import CrackDataset, SegmentationPresetTrain, SegmentationPresetEval
from train_utils.train_and_eval import evaluate
from models.segformer.segformer import SegFormer
from models.unet.unet import UNet
from models.unet.mobilenet_unet import MobileV3Unet
from models.unet.vgg_unet import VGG16UNet
from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
from models.fcn.fcn import fcn_resnet50
from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
from models.unet.UnetPP import UNetPP



def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def evaluate1(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dices = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_time = 0.
    with torch.no_grad():
        init_img = torch.zeros((1, 3, 512, 512), device=device)
        model(init_img)

        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            t_start = time_synchronized()
            output = model(image)
            t_end = time_synchronized()
            total_time += t_end - t_start

            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())

            pred = F.one_hot(output.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
            dice_target = build_target(target, num_classes, 255)
            dice = multiclass_dice_coeff(pred[:, 1:], dice_target[:, 1:], 255)
            dices.append(dice.item())

        confmat.reduce_from_all_processes()

    return confmat, dices, total_time


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # assert os.path.exists(args.weights), f"weights {args.weights} not found."

    num_classes = args.num_classes + 1
    img_size = (512, 512)  # Set the input size for the model
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_dataset = CrackDataset(args.data_path, train=False,
                               transforms=SegmentationPresetEval(img_size, mean=mean, std=std))

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=args.batch_size,  # must be 1
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 collate_fn=val_dataset.collate_fn)

    # model = UNet(in_channels=3, num_classes=num_classes, base_c=64)
    # model = VGG16UNet(num_classes=num_classes)
    # model = MobileV3Unet(num_classes=num_classes)
    # model = SegFormer(num_classes=num_classes, phi=args.phi)
    # pretrain_weights = torch.load(args.weights, map_location='cpu')
    # if "model" in pretrain_weights:
    #     model.load_state_dict(pretrain_weights["model"])
    # else:
    #     model.load_state_dict(pretrain_weights)
    # model.to(device)

    weight_paths = {
        "SegFormer_b0": r"D:\Devendra_Files\CrackSegFormer-main\weights\Segformer\20250806-011324-best_model.pth",
        "SegFormer_b5": r"D:\Devendra_Files\CrackSegFormer-main\weights\Segformer_b5/Segformer_b5__best_epoch10_dice0.542.pth",
        "UNetPP": r"D:\Devendra_Files\CrackSegFormer-main\weights\UNetPP_1\UnetPP_1_best_epoch89_dice0.574.pth",
        "DeepLabV3": r"D:\Devendra_Files\CrackSegFormer-main\weights\DeepLab\DeepLab_best_epoch2_dice0.425.pth",
    }
    models = {
        "SegFormer_b0": SegFormer(num_classes=num_classes, phi='b0', pretrained=False),
        "SegFormer_b5": SegFormer(num_classes=num_classes, phi='b5', pretrained=False),
        "UNetPP": UNetPP(in_channels=3, num_classes=num_classes),
        "DeepLabV3": deeplabv3_resnet101(aux=True, num_classes=num_classes, pretrain_backbone=False)
    }

    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        model.to(device)
        weight_path = weight_paths.get(name, None)
        if weight_path and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device)
            # Adjust loading if checkpoint contains keys under 'model'
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: weight for {name} not found, using default initialized weights!")

        confmat, dices,  = evaluate(model, val_loader, device=device, num_classes=num_classes)
        mean_dice = np.mean(dices)
        std_dice = np.std(dices)
        results[name] = {
            "confmat": confmat,
            "mean_dice": mean_dice,
            "std_dice": std_dice,
            # "total_time": total_time,
            # "time_per_image": total_time / len(val_dataset)
        }
        print(confmat)
        print(f"Model {name} - mean Dice: {mean_dice:.3f}, std Dice: {std_dice:.3f}") #, Total time: {total_time:.2f}s")
        print("" + "=" * 50)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="validation")
    parser.add_argument("--device", default="cuda:0", help="training device")

    parser.add_argument("--data-path",
                        default=r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_SPLIT_OLD",
                        help="images root")
    parser.add_argument("--num-classes", default=1, type=int)  # exclude background

    parser.add_argument("--phi", default="b0", help="Use backbone")
    parser.add_argument("--weights", default="")


    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
