"""train_utils/utils.py"""
import cv2
import numpy as np


def show_config(config):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in config.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


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


import cv2
import numpy as np


def plot(
        train_loss,
        dice_scores,
        save_path,
        width=1600,
        height=900):

    if len(train_loss) == 0:
        return

    canvas = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8
    )

    margin_left = 100
    margin_right = 50
    margin_top = 80
    margin_bottom = 80

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # -----------------------------
    # Data
    # -----------------------------
    epochs = np.arange(1, len(train_loss) + 1)

    loss_arr = np.array(train_loss, dtype=np.float32)
    dice_arr = np.array(dice_scores, dtype=np.float32)

    y_min = min(loss_arr.min(), dice_arr.min())
    y_max = max(loss_arr.max(), dice_arr.max())

    padding = (y_max - y_min) * 0.10
    y_min -= padding
    y_max += padding

    if y_max == y_min:
        y_max += 1.0

    # -----------------------------
    # Draw Grid
    # -----------------------------
    num_y_ticks = 8

    for i in range(num_y_ticks + 1):

        y_val = y_min + (y_max - y_min) * i / num_y_ticks

        y = int(
            height - margin_bottom -
            (y_val - y_min) / (y_max - y_min) * plot_h
        )

        cv2.line(
            canvas,
            (margin_left, y),
            (width - margin_right, y),
            (220, 220, 220),
            1
        )

        cv2.putText(
            canvas,
            f"{y_val:.2f}",
            (15, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    # -----------------------------
    # X Axis
    # -----------------------------
    num_x_ticks = min(10, len(epochs))

    for i in range(num_x_ticks + 1):

        idx = int(i * (len(epochs) - 1) / max(num_x_ticks, 1))

        x = int(
            margin_left +
            idx / max(len(epochs) - 1, 1) * plot_w
        )

        cv2.line(
            canvas,
            (x, margin_top),
            (x, height - margin_bottom),
            (230, 230, 230),
            1
        )

        cv2.putText(
            canvas,
            str(idx + 1),
            (x - 10, height - margin_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    # -----------------------------
    # Axes
    # -----------------------------
    cv2.line(
        canvas,
        (margin_left, margin_top),
        (margin_left, height - margin_bottom),
        (0, 0, 0),
        2
    )

    cv2.line(
        canvas,
        (margin_left, height - margin_bottom),
        (width - margin_right, height - margin_bottom),
        (0, 0, 0),
        2
    )

    # -----------------------------
    # Convert points
    # -----------------------------
    def convert_points(values):

        pts = []

        for i, val in enumerate(values):

            x = int(
                margin_left +
                i / max(len(values) - 1, 1) * plot_w
            )

            y = int(
                height - margin_bottom -
                (val - y_min) / (y_max - y_min) * plot_h
            )

            pts.append((x, y))

        return np.array(pts, dtype=np.int32)

    loss_pts = convert_points(loss_arr)
    dice_pts = convert_points(dice_arr)

    # -----------------------------
    # Draw Curves
    # -----------------------------
    cv2.polylines(
        canvas,
        [loss_pts],
        False,
        (255, 0, 0),
        3
    )

    cv2.polylines(
        canvas,
        [dice_pts],
        False,
        (0, 180, 0),
        3
    )

    # -----------------------------
    # Draw Last Points
    # -----------------------------
    cv2.circle(
        canvas,
        tuple(loss_pts[-1]),
        6,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        canvas,
        tuple(dice_pts[-1]),
        6,
        (0, 180, 0),
        -1
    )

    # -----------------------------
    # Title
    # -----------------------------
    cv2.putText(
        canvas,
        "Training Curves",
        (60, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # -----------------------------
    # Statistics
    # -----------------------------
    current_loss = float(loss_arr[-1])
    current_dice = float(dice_arr[-1])
    best_dice = float(dice_arr.max())

    stats_x = width - 400

    cv2.putText(
        canvas,
        f"Epoch: {len(train_loss)}",
        (stats_x, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Loss: {current_loss:.4f}",
        (stats_x, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Dice: {current_dice:.4f}",
        (stats_x, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Best Dice: {best_dice:.4f}",
        (stats_x, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 0),
        2
    )

    # -----------------------------
    # Legend
    # -----------------------------
    legend_y = 40

    cv2.line(
        canvas,
        (width - 320, legend_y),
        (width - 260, legend_y),
        (255, 0, 0),
        3
    )

    cv2.putText(
        canvas,
        "Loss",
        (width - 245, legend_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv2.line(
        canvas,
        (width - 320, legend_y + 40),
        (width - 260, legend_y + 40),
        (0, 180, 0),
        3
    )

    cv2.putText(
        canvas,
        "Dice",
        (width - 245, legend_y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    # -----------------------------
    # Axis labels
    # -----------------------------
    cv2.putText(
        canvas,
        "Epoch",
        (width // 2 - 40, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv2.imwrite(save_path, canvas)