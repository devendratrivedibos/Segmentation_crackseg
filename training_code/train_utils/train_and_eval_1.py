import torch
from torch import nn
from . import distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W)
        # targets: (N, H, W)
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)  # p_t = e^(-CE)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def criterion(inputs, target, loss_weight=None, num_classes: int = 2,
              use_ce: bool = True, use_focal: bool = False, use_dice: bool = True,
              focal_gamma: float = 2.0, ignore_index: int = -100):
    """
    Flexible criterion to enable/disable CrossEntropy, FocalLoss, and DiceLoss.
    """
    losses = {}
    focal_loss_fn = FocalLoss(alpha=loss_weight, gamma=focal_gamma, ignore_index=ignore_index)

    for name, x in inputs.items():
        total_loss = 0.0

        # CrossEntropy Loss
        if use_ce:
            ce_loss = nn.functional.cross_entropy(
                x, target,
                ignore_index=ignore_index,
                weight=loss_weight
            )
            total_loss += ce_loss

        # Focal Loss
        if use_focal:
            fl = focal_loss_fn(x, target)
            total_loss += fl

        # Dice Loss
        if use_dice:
            dice_target = build_target(target, num_classes, ignore_index)
            dl = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
            total_loss += dl

        losses[name] = total_loss

    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']



def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes, lr_scheduler,
                    print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 1.0], device=device)
    else:
        # Example: adjust weights according to dataset
        loss_weight = torch.tensor([
            0.3,  # Background
            1.0,  # Alligator
            1.2,  # Transverse
            1.2,  # Longitudinal
            1.2,  # Multiple
            5.0   # Joint Seal
        ], dtype=torch.float32).to(device)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes,
                            use_ce=False, use_focal=True, use_dice=True,
                            ignore_index=255, focal_gamma=2.0)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


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
