import os
import cv2
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import re
import random

# --- CONFIG ---
start_number = 0  # <<< starting image number
root_dir = "D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_ASPHALT_OLD\REDUCED_DATASET_SPLIT\TEST"
# root_dir = "D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\DATA\REDUCED_DATASET_SPLIT\TEST"
image_dir = os.path.join(root_dir, 'IMAGES')
orig_mask_dir = os.path.join(root_dir, 'MASKS')
pred_mask_dir = os.path.join(root_dir, 'RESULTS')

accepted_img_dir = os.path.join(root_dir, "ACCEPTED_IMAGES")
accepted_mask_dir = os.path.join(root_dir, "ACCEPTED_MASKS")
rework_img_dir = os.path.join(root_dir, "REWORK_IMAGES")
rework_mask_dir = os.path.join(root_dir, "REWORK_MASKS")

# Ensure dirs exist
for d in [accepted_img_dir, accepted_mask_dir, rework_img_dir, rework_mask_dir]:
    os.makedirs(d, exist_ok=True)

# --- Load file names ---
images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
orig_masks = sorted([f for f in os.listdir(orig_mask_dir) if f.lower().endswith('.png')])
pred_masks = sorted([f for f in os.listdir(pred_mask_dir) if f.lower().endswith('.png')])

# --- Strip extensions for matching ---
image_stems = {os.path.splitext(f)[0]: f for f in images}
orig_mask_stems = {os.path.splitext(f)[0]: f for f in orig_masks}
pred_mask_stems = {os.path.splitext(f)[0]: f for f in pred_masks}

# --- Keep only intersection across all three ---
common_keys = sorted(set(image_stems) & set(orig_mask_stems) & set(pred_mask_stems))

images = [image_stems[k] for k in common_keys]
orig_masks = [orig_mask_stems[k] for k in common_keys]
pred_masks = [pred_mask_stems[k] for k in common_keys]

print(f"Kept {len(images)} triplets out of "
      f"{len(image_stems)} images, {len(orig_mask_stems)} orig_masks, {len(pred_mask_stems)} pred_masks")

assert len(images) == len(orig_masks) == len(pred_masks), "Images and masks count mismatch!"

# --- Shuffle together ---
combined = list(zip(images, orig_masks, pred_masks))
random.shuffle(combined)  # Shuffle once
images, orig_masks, pred_masks = zip(*combined)
images, orig_masks, pred_masks = list(images), list(orig_masks), list(pred_masks)


def find_start_index(files, number):
    for i, f in enumerate(files):
        if re.search(rf"{number}", f):
            return i
    return 0


# --- Viewer Class ---
class ImageMaskViewer:
    def __init__(self, images, orig_masks, pred_masks, image_dir, orig_mask_dir, pred_mask_dir, start_index=0):
        self.images = images
        self.orig_masks = orig_masks
        self.pred_masks = pred_masks
        self.image_dir = image_dir
        self.orig_mask_dir = orig_mask_dir
        self.pred_mask_dir = pred_mask_dir
        self.index = start_index

        # Create 4 columns: Image | Original Mask | Predicted Mask | Overlay(pred)
        self.fig, self.axs = plt.subplots(1, 4, figsize=(20, 5))
        plt.subplots_adjust(bottom=0.2)

        # Buttons
        axprev = plt.axes([0.2, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.35, 0.05, 0.1, 0.075])
        axaccept = plt.axes([0.55, 0.05, 0.1, 0.075])
        axrework = plt.axes([0.7, 0.05, 0.1, 0.075])

        self.bprev = Button(axprev, 'Previous')
        self.bnext = Button(axnext, 'Next')
        self.baccept = Button(axaccept, 'Accept')
        self.brework = Button(axrework, 'Rework')

        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)
        self.baccept.on_clicked(self.accept)
        self.brework.on_clicked(self.rework)

        self.update_display()
        plt.show()

    def load_images(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        orig_mask_path = os.path.join(self.orig_mask_dir, self.orig_masks[idx])
        pred_mask_path = os.path.join(self.pred_mask_dir, self.pred_masks[idx])

        # Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Original mask
        orig_mask = cv2.imread(orig_mask_path, cv2.IMREAD_UNCHANGED)
        if len(orig_mask.shape) == 2:
            orig_mask_colored = cv2.applyColorMap(orig_mask, cv2.COLORMAP_JET)
        else:
            orig_mask_colored = cv2.cvtColor(orig_mask, cv2.COLOR_BGR2RGB)

        # Predicted mask
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
        if len(pred_mask.shape) == 2:
            pred_mask_colored = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
        else:
            pred_mask_colored = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB)

        # Overlay using predicted mask
        overlay = cv2.addWeighted(img, 0.5, pred_mask_colored, 0.5, 0)

        return img, orig_mask_colored, pred_mask_colored, overlay

    def update_display(self):
        img, orig_mask, pred_mask, overlay = self.load_images(self.index)

        self.axs[0].imshow(img)
        self.axs[0].set_title("Image")
        self.axs[0].axis("off")

        self.axs[1].imshow(orig_mask)
        self.axs[1].set_title("Original Mask")
        self.axs[1].axis("off")

        self.axs[2].imshow(pred_mask)
        self.axs[2].set_title("Predicted Mask")
        self.axs[2].axis("off")

        self.axs[3].imshow(overlay)
        self.axs[3].set_title("Overlay (Pred)")
        self.axs[3].axis("off")

        self.fig.suptitle(f"[{self.index + 1}/{len(self.images)}] {self.images[self.index]}")
        self.fig.canvas.draw()

    def move_files(self, dest_img_dir, dest_mask_dir):
        img_name = self.images[self.index]
        mask_name = self.pred_masks[self.index]

        src_img = os.path.join(self.image_dir, img_name)
        src_mask = os.path.join(self.pred_mask_dir, mask_name)

        dst_img = os.path.join(dest_img_dir, img_name)
        dst_mask = os.path.join(dest_mask_dir, mask_name)

        shutil.move(src_img, dst_img)
        shutil.move(src_mask, dst_mask)

        print(f"Moved {img_name} and {mask_name} -> {dest_img_dir}, {dest_mask_dir}")

    def accept(self, event):
        self.move_files(accepted_img_dir, accepted_mask_dir)
        self.next(event)

    def rework(self, event):
        self.move_files(rework_img_dir, rework_mask_dir)
        self.next(event)

    def next(self, event):
        self.index = (self.index + 1) % len(self.images)
        self.update_display()

    def prev(self, event):
        self.index = (self.index - 1) % len(self.images)
        self.update_display()


# --- Find start index ---
start_index = find_start_index(images, start_number)

# --- Run Viewer ---
viewer = ImageMaskViewer(images, orig_masks, pred_masks, image_dir, orig_mask_dir, pred_mask_dir,
                         start_index=start_index)
