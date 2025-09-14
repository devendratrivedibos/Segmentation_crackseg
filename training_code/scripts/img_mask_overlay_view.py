import os
import cv2
import shutil
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import re

# --- CONFIG ---
root_dir = r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3"
image_dir = os.path.join(root_dir, 'AnnotationImages')
mask_dir = os.path.join(root_dir, 'AnnotationMasks')
start_number = 0   # <<< put the number you want to start with


# Output dirs
accepted_img_dir = os.path.join(root_dir, "ACCEPTED_IMAGES")
accepted_mask_dir = os.path.join(root_dir, "ACCEPTED_MASKS")
rework_img_dir = os.path.join(root_dir, "REWORK_IMAGES")
rework_mask_dir = os.path.join(root_dir, "REWORK_MASKS")
csv_log = os.path.join(root_dir, "rework_log.csv")

# Make sure dirs exist
for d in [accepted_img_dir, accepted_mask_dir, rework_img_dir, rework_mask_dir]:
    os.makedirs(d, exist_ok=True)

# --- Load file names ---
images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
masks  = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])

# --- Strip extensions for matching ---
image_stems = {os.path.splitext(f)[0]: f for f in images}
mask_stems = {os.path.splitext(f)[0]: f for f in masks}

# --- Keep only intersection ---
common_keys = sorted(set(image_stems.keys()) & set(mask_stems.keys()))
images = [image_stems[k] for k in common_keys]
masks = [mask_stems[k]  for k in common_keys]


assert len(images) == len(masks), "Images and masks count mismatch!"

def find_start_index(files, number):
    """Find index of file containing the given number."""
    for i, f in enumerate(files):
        if re.search(rf"{number}", f):  # match number inside filename
            return i
    return 0  # fallback to first image if not found

# --- Find start index ---
start_index = find_start_index(images, start_number)

# --- Viewer Class ---
class ImageMaskViewer:
    def __init__(self, images, masks, image_dir, mask_dir, start_index=0):
        self.images = images
        self.masks = masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.index = start_index
        self.comment = ""

        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.25)

        # Buttons
        axprev   = plt.axes([0.15, 0.15, 0.1, 0.07])
        axnext   = plt.axes([0.30, 0.15, 0.1, 0.07])
        axaccept = plt.axes([0.55, 0.15, 0.12, 0.07])
        axrework = plt.axes([0.70, 0.15, 0.12, 0.07])
        axbox    = plt.axes([0.15, 0.05, 0.67, 0.05])  # text box

        self.bnext   = Button(axnext, 'Next')
        self.bprev   = Button(axprev, 'Previous')
        self.baccept = Button(axaccept, 'Accept')
        self.brework = Button(axrework, 'Rework')
        self.textbox = TextBox(axbox, "Comment: ")

        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)
        self.baccept.on_clicked(self.accept)
        self.brework.on_clicked(self.rework)
        self.textbox.on_submit(self.update_text)

        self.update_display()
        plt.show()

    def load_images(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 2:
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        else:
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img, 0.5, mask_colored, 0.5, 0)
        return img, mask_colored, overlay

    def update_display(self):
        if not self.images:
            self.fig.suptitle("No more images left!", fontsize=14, color="red")
            for ax in self.axs:
                ax.clear()
                ax.axis("off")
            self.fig.canvas.draw()
            return

        img, mask, overlay = self.load_images(self.index)

        self.axs[0].imshow(img);     self.axs[0].set_title("Image");   self.axs[0].axis("off")
        self.axs[1].imshow(mask);    self.axs[1].set_title("Mask");    self.axs[1].axis("off")
        self.axs[2].imshow(overlay); self.axs[2].set_title("Overlay"); self.axs[2].axis("off")

        self.fig.suptitle(f"Image {self.index+1}/{len(self.images)}: {self.images[self.index]}")
        self.fig.canvas.draw()

    def update_text(self, text):
        self.comment = text  # store current comment

    def move_files(self, target_img_dir, target_mask_dir, log_comment=False):
        img_name  = self.images[self.index]
        mask_name = self.masks[self.index]

        src_img  = os.path.join(self.image_dir, img_name)
        src_mask = os.path.join(self.mask_dir, mask_name)
        dst_img  = os.path.join(target_img_dir, img_name)
        dst_mask = os.path.join(target_mask_dir, mask_name)

        shutil.move(src_img, dst_img)
        shutil.move(src_mask, dst_mask)

        # Log if rework
        if log_comment:
            with open(csv_log, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([img_name, self.comment])
            print(f"Logged: {img_name} | {self.comment}")

        # Remove from list
        del self.images[self.index]
        del self.masks[self.index]

        # Adjust index
        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        self.comment = ""
        self.textbox.set_val("")
        self.update_display()

    def accept(self, event):
        if self.images:
            self.move_files(accepted_img_dir, accepted_mask_dir, log_comment=False)

    def rework(self, event):
        if self.images:
            self.move_files(rework_img_dir, rework_mask_dir, log_comment=True)

    def next(self, event):
        if self.images:
            self.index = (self.index + 1) % len(self.images)
            self.update_display()

    def prev(self, event):
        if self.images:
            self.index = (self.index - 1) % len(self.images)
            self.update_display()


# --- Run Viewer ---
viewer = ImageMaskViewer(images, masks, image_dir, mask_dir, start_index=start_index)
