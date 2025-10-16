"""

Distributed Data Parallel (DDP) training script for UNet++ model using PyTorch.

python train_ddp_unetpp.py --nodes 2 --nr 0 --master-addr "192.168.1.10" --master-port 29500 --gpus 2

"""


import os
import sys
import time
import datetime
import argparse
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SubsetRandomSampler, DistributedSampler

import numpy as np
import cv2

# Import local modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))
from models.unet.UnetPP import UNetPP
from train_utils.train_and_eval_1 import evaluate, train_one_epoch_loss, create_lr_scheduler
from train_utils.my_dataset_asphalt import CrackDataset, SegmentationPresetTrain, SegmentationPresetEval


# -----------------------------
# Distributed Initialization
# -----------------------------
def init_distributed(rank, world_size, master_addr="127.0.0.1", master_port="29500"):
    """Initialize torch.distributed safely for Windows/Linux."""
    if sys.platform == "win32":
        backend = "gloo"  # NCCL unsupported on Windows
        store_dir = os.path.join(tempfile.gettempdir(), "torch_ddp_store")
        os.makedirs(store_dir, exist_ok=True)
        init_method = f"file://{store_dir}/shared_init"
    else:
        backend = "nccl"
        init_method = f"tcp://{master_addr}:{master_port}"

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    print(f"[Rank {rank}] ✅ DDP initialized with backend={backend}, world_size={world_size}")


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="UNet++ Distributed Training")
    # DDP distributed setup
    parser.add_argument("--nodes", default=2, type=int)
    parser.add_argument("--gpus", default=torch.cuda.device_count(), type=int)
    parser.add_argument("--nr", default=0, type=int, help="Node rank (0 for master node)")
    parser.add_argument("--master-addr", default="192.168.1.16")
    parser.add_argument("--master-port", default="29500")

    # Training config
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-path", default=r"Z:/Devendra/ASPHALT_ACCEPTED/SPLIT")
    parser.add_argument("--num-classes", default=5, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--optimizer-type", default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("--pretrained-weights",
                        default=r"X:/Devendra_Files/CrackSegFormer-main/weights/UNET_asphalt_1024/UNET_best.pth")
    parser.add_argument("--save-dir", default="./weights/UNETPP_DDP", type=str)
    parser.add_argument("--resume", default='', help="resume checkpoint path")
    parser.add_argument("--amp", default=True, type=bool)
    parser.add_argument("--verbose", default=True, type=bool)
    return parser.parse_args()


# -----------------------------
# Dataset/Model Utilities
# -----------------------------
def get_transform(train, img_size_h=1024, img_size_w=1024, mean=(0.488, 0.488, 0.488), std=(0.149, 0.149, 0.149)):
    if train:
        return SegmentationPresetTrain(img_size_h, img_size_w, mean=mean, std=std)
    else:
        return SegmentationPresetEval(img_size_h, img_size_w, mean=mean, std=std)


def create_model(num_classes):
    model = UNetPP(in_channels=3, num_classes=num_classes)
    return model


# -----------------------------
# Main Training Function
# -----------------------------
def train_ddp(rank, world_size, args):
    # Init DDP
    init_distributed(rank, world_size, args.master_addr, args.master_port)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Dataset setup
    train_dataset = CrackDataset(args.data_path, train=True,
                                 transforms=get_transform(train=True))
    val_dataset = CrackDataset(args.data_path, train=False,
                               transforms=get_transform(train=False))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
                            num_workers=2, pin_memory=True, collate_fn=val_dataset.collate_fn)

    # Model setup
    model = create_model(args.num_classes + 1).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Optimizer
    if args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume checkpoint
    best_dice = 0.0
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_dice = checkpoint.get("best_dice", 0.0)
        print(f"[Rank {rank}] Resumed from epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)
    results_file = Path(args.save_dir) / f"rank{rank}_results.txt"

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=10)
        epoch_loss, lr = train_one_epoch_loss(model, optimizer, train_loader, device, epoch,
                                              num_classes=args.num_classes + 1, output_dir=args.save_dir,
                                              lr_scheduler=lr_scheduler, print_freq=10, scaler=scaler)

        confmat, dice_score = evaluate(model, val_loader, device, num_classes=args.num_classes + 1)

        if rank == 0:
            print(f"[Epoch {epoch}] loss={epoch_loss:.4f}, dice={dice_score:.4f}")
            with open(results_file, "a") as f:
                f.write(f"[epoch: {epoch}] loss={epoch_loss:.4f}, dice={dice_score:.4f}\n")

            # Save checkpoint
            ckpt = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "dice_score": dice_score
            }
            torch.save(ckpt, f"{args.save_dir}/UNETPP_epoch_{epoch}_rank0.pth")

            if dice_score > best_dice:
                best_dice = dice_score
                torch.save(ckpt, f"{args.save_dir}/UNETPP_best.pth")
                print(f"✅ Best model saved at epoch {epoch} (dice={best_dice:.4f})")

    dist.destroy_process_group()


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    world_size = args.gpus * args.nodes
    print(f"🚀 Starting DDP with {args.nodes} nodes × {args.gpus} GPUs = {world_size} processes")
    mp.spawn(train_ddp, nprocs=args.gpus, args=(world_size, args))


