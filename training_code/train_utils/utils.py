"""train_utils/utils.py"""
import scipy.signal
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import datetime


def show_config(config):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in config.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def plot(train_loss,
         dice_scores,
         save_path):

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))

    # Raw curves
    plt.plot(
        epochs,
        train_loss,
        linewidth=2,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        dice_scores,
        linewidth=2,
        label="Val Dice"
    )

    # Smoothed curves
    if len(train_loss) >= 5:
        try:
            window = min(
                len(train_loss) if len(train_loss) % 2 == 1 else len(train_loss) - 1,
                11
            )

            if window >= 5:

                smooth_loss = scipy.signal.savgol_filter(
                    train_loss,
                    window_length=window,
                    polyorder=3
                )

                smooth_dice = scipy.signal.savgol_filter(
                    dice_scores,
                    window_length=window,
                    polyorder=3
                )

                plt.plot(
                    epochs,
                    smooth_loss,
                    linestyle="--",
                    linewidth=2,
                    label="Smooth Loss"
                )

                plt.plot(
                    epochs,
                    smooth_dice,
                    linestyle="--",
                    linewidth=2,
                    label="Smooth Dice"
                )

        except Exception as e:
            print(f"Plot smoothing error: {e}")

    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Progress")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def show_label(label):
    img = label.convert('RGBA')
    x, y = img.size
    for i in range(x):
        for j in range(y):
            color = img.getpixel((i, j))
            Mean = np.mean(list(color[:-1]))
            if Mean < 255:
                color = color[:-1] + (0,)
            else:
                color = (255, 97, 0, 255)
            img.putpixel((i, j), color)
    return img
