import torch
from torch import nn
from typing import Union, Dict

from . import distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target   # kept for compatibility if you use elsewhere
from .hybrid_loss import build_crack_criterion               # your hybrid criterion builder


# -----------------------------
# Global config
# -----------------------------
IGNORE_INDEX = 255

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 13

class_counts = None

# Build the hybrid loss ONCE (don’t recreate inside the loop)
criterion = build_crack_criterion(
    num_classes=num_classes,
    class_counts=class_counts,
    device=device,
    ignore_index=IGNORE_INDEX
)


# -----------------------------
# Helpers
# -----------------------------
def _main_output(outputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Return the primary logits tensor [B, C, H, W] from either a Tensor or a dict with 'out'.
    """
    if isinstance(outputs, dict):
        assert "out" in outputs, "Model dict output must contain key 'out'."
        return outputs["out"]
    return outputs


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=IGNORE_INDEX)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 50, header):
            image, target = image.to(device), target.to(device)

            outputs = model(image)                 # Tensor or dict
            logits = _main_output(outputs)         # [B, C, H, W]
            pred = logits.argmax(1)                # [B, H, W]

            confmat.update(target.flatten(), pred.flatten())
            dice.update(logits, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


# -----------------------------
# Training (one epoch)
# -----------------------------
def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    num_classes,
                    lr_scheduler,
                    print_freq=10,
                    scaler=None,
                    grad_clip_norm: float = 0.0):
    """
    Trains one epoch using the global `criterion` built above (hybrid crack loss).
    - Supports Tensor or dict outputs (with optional 'aux')
    - Optional gradient clipping via grad_clip_norm > 0
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(image)        # Tensor or {'out', 'aux'}
            loss = criterion(outputs, target)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


# -----------------------------
# LR Scheduler (unchanged)
# -----------------------------
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) /
                    ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
