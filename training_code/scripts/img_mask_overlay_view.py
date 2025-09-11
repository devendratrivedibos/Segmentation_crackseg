import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import re

# --- CONFIG ---
image_dir = r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3\AnnotationImages"
mask_dir = r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3\AnnotationMasks"
start_number = 82   # <<< put the number you want to start with

# --- Load file names ---
images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])

assert len(images) == len(masks), "Images and masks count mismatch!"

def find_start_index(files, number):
    """Find index of file containing the given number."""
    for i, f in enumerate(files):
        if re.search(rf"{number}", f):  # match number inside filename
            return i
    return 0  # fallback to first image if not found

# --- Viewer Class ---
class ImageMaskViewer:
    def __init__(self, images, masks, image_dir, mask_dir, start_index=0):
        self.images = images
        self.masks = masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.index = start_index

        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.2)

        # Buttons
        axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bprev = Button(axprev, 'Previous')
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)

        self.update_display()
        plt.show()

    def load_images(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 2:
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)179
        else:
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img, 0.5, mask_colored, 0.5, 0)
        return img, mask_colored, overlay

    def update_display(self):
        img, mask, overlay = self.load_images(self.index)

        self.axs[0].imshow(img)
        self.axs[0].set_title("Image")
        self.axs[0].axis("off")

        self.axs[1].imshow(mask)
        self.axs[1].set_title("Mask")
        self.axs[1].axis("off")

        self.axs[2].imshow(overlay)
        self.axs[2].set_title("Overlay")
        self.axs[2].axis("off")

        self.fig.suptitle(f"Image {self.index+1}/{len(self.images)}: {self.images[self.index]}")
        self.fig.canvas.draw()

    def next(self, event):
        self.index = (self.index + 1) % len(self.images)
        self.update_display()

    def prev(self, event):
        self.index = (self.index - 1) % len(self.images)
        self.update_display()

# --- Find start index ---
start_index = find_start_index(images, start_number)

# --- Run Viewer ---
viewer = ImageMaskViewer(images, masks, image_dir, mask_dir, start_index=start_index)
