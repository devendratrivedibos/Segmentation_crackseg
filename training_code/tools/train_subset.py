import os
import pdb
import time
import datetime
import torch
import numpy as np
import sys
import cv2
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))
from models.unet.UnetPP import UNetPP
from pathlib import Path

from torch.utils.data import DataLoader, SubsetRandomSampler
from train_utils.train_and_eval_1 import evaluate, train_one_epoch_loss, create_lr_scheduler
from train_utils.my_dataset_subset import CrackDataset, SegmentationPresetTrain, SegmentationPresetEval
import train_utils.transforms as T
from train_utils.utils import plot, show_config

# Get project root (parent of tools/)
project_root_ = Path(__file__).resolve().parent.parent.parent
OUTPUT_SAVE_PATH = project_root_ / 'weights' / 'UNET_4oct'  # Change this to your desired output path
model_name = "UNET_4oct"
os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
results_file = OUTPUT_SAVE_PATH / "{}-results.txt".format(model_name)


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch unet training (subset loader + iter checkpoints)")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("--data-path", default=r"G:/Devendra/ASPHALT_ACCEPTED/SPLIT", help="root")
    parser.add_argument("--num-classes", default=5, type=int)  # exclude background
    parser.add_argument("--use-subset", default=False, type=bool, help="use random subset per epoch")
    parser.add_argument("--aux", default=True, type=bool, help="aux loss if any")
    parser.add_argument("--pretrained", default=True, type=bool, help="backbone pretrained")
    parser.add_argument("--pretrained-weights", type=str,
                        default=r"W:\Devendra_Files\CrackSegFormer-main\weights\27Sept_Asphalt\27Sept_Asphalt_best_epoch267_dice0.742.pth",
                        help="pretrained weights path")
    parser.add_argument("--optimizer-type", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--subset-size", default=30000, type=int, help="random subset size per epoch")
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--print-freq", default=1, type=int)
    parser.add_argument("--warmup-epochs", default=10, type=int)
    parser.add_argument("--save-best", default=True, type=bool, help="save best epoch")
    parser.add_argument("--save-every-iter", default=100, type=int,
                        help="if 1 -> save latest iter every iteration (overwrites); if >1 save/overwrite every N iters")
    parser.add_argument("--resume", default='', help="resume from checkpoint path")
    parser.add_argument("--amp", default=True, type=bool, help="use mixed precision")
    parser.add_argument("--verbose", default=True, type=bool, help="log checkpoint saves")
    args = parser.parse_args()
    return args


def get_subset_loader(dataset, batch_size, subset_size, num_workers):
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=dataset.collate_fn)
    return loader


def get_transform(train, img_size_h=5120, img_size_w=2560, mean=(0.541, 0.541, 0.541), std=(0.144, 0.144, 0.144)):
    if train:
        return SegmentationPresetTrain(img_size_h, img_size_w, mean=mean, std=std)
    else:
        return SegmentationPresetEval(img_size_h, img_size_w, mean=mean, std=std)


def create_model(num_classes, pretrained=True):
    # model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)
    # model = fcn_resnet50(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)
    # model = deeplabv3_resnet101(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)
    # model = deeplabv3_mobilenetv3_large(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)
    # model = SegFormer(num_classes=num_classes, phi=args.phi, pretrained=args.pretrained)
    # model = UNet(in_channels=3, num_classes=num_classes, base_c=64)
    # model = MobileV3Unet(num_classes=num_classes, pretrain_backbone=args.pretrained)
    # model = VGG16UNet(num_classes=num_classes, pretrain_backbone=args.pretrained)
    # model = DINODeepLab(num_classes=num_classes, backbone_name="dinov2_vitl14")
    model = UNetPP(in_channels=3, num_classes=num_classes)
    return model


def main(args):
    device = torch.device(args.device)
    mean = (0.488, 0.488, 0.488)
    std = (0.149, 0.149, 0.149)
    img_size_h, img_size_w = 1024, 1024
    # Dataset
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_dataset = CrackDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, img_size_h=img_size_h, img_size_w=img_size_w,
                                                          mean=mean, std=std))

    val_dataset = CrackDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, img_size_h=img_size_h, img_size_w=img_size_w,
                                                        mean=mean, std=std))

    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Model
    model = create_model(num_classes=args.num_classes + 1, pretrained=args.pretrained)  # +1 for background
    model.to(device)

    # Optimizer
    if args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.pqarameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = args.start_epoch
    best_dice = 0.0

    # Resume if checkpoint
    if args.pretrained and os.path.isfile(args.pretrained_weights):
        print(f"=> loading pretrained checkpoint {args.pretrained_weights}")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
    # Resume if checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_dice = checkpoint.get("best_dice", 0.0)
        print(f"=> resumed at epoch {start_epoch}, best dice: {best_dice:.4f}")

    global_iter = 0

    for epoch in range(start_epoch, args.epochs):
        if args.use_subset:
            train_loader = get_subset_loader(train_dataset, args.batch_size, args.subset_size, num_workers=num_workers)
            val_loader = get_subset_loader(val_dataset, args.batch_size, args.subset_size // 10,
                                           num_workers=num_workers)

        else:
            # Full dataset loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True, num_workers=num_workers,
                pin_memory=True, collate_fn=train_dataset.collate_fn)
            val_loader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=val_dataset.collate_fn)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=10)
        epoch_loss, lr = train_one_epoch_loss(model, optimizer, train_loader, device, epoch,
                                              num_classes=args.num_classes + 1, output_dir=OUTPUT_SAVE_PATH,
                                              lr_scheduler=lr_scheduler, print_freq=1, scaler=scaler)

        confmat, dice_score = evaluate(model, val_loader, device, num_classes=args.num_classes + 1)
        val_info = str(confmat)

        print(f"Epoch {epoch} finished: mean_loss={epoch_loss:.4f}, dice={dice_score:.4f} val_info:\n{val_info}")
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n"
            f.write(train_info + val_info + "\n\n")
        # Save checkpoint per epoch
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "dice_score": dice_score
        }, f"{OUTPUT_SAVE_PATH}/{model_name}_epoch_{epoch}_dice_{dice_score:.4f}.pth")

        # Update best
        if dice_score > best_dice:
            best_dice = dice_score
            if args.save_best:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "dice_score": dice_score
                }, f"{OUTPUT_SAVE_PATH}/{model_name}_best.pth")
                if args.verbose:
                    print(f"✔ Best model saved at epoch {epoch} with dice {best_dice:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
