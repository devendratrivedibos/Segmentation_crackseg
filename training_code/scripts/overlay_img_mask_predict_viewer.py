import os
import cv2
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import re
import random
from collections import OrderedDict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import gc
import csv

# --- CONFIG ---
start_number = 0  # <<< starting image number
root_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_OG_ACCEPTED"
image_dir = os.path.join(root_dir, 'DATASET_IMAGES_CLASS_4')
orig_mask_dir = os.path.join(root_dir, 'DATASET_MASKS_CLASS_4')
pred_mask_dir = os.path.join(root_dir, 'DATASET_RESULTS_CLASS_4')

accepted_img_dir = os.path.join(root_dir, "ACCEPTED_IMAGES")
accepted_mask_dir = os.path.join(root_dir, "ACCEPTED_MASKS")
rework_img_dir = os.path.join(root_dir, "REWORK_IMAGES")
rework_mask_dir = os.path.join(root_dir, "REWORK_MASKS")
csv_log = os.path.join(root_dir, "review_log.csv")

# Ensure dirs exist
# for d in [accepted_img_dir, accepted_mask_dir, rework_img_dir, rework_mask_dir]:
#     os.makedirs(d, exist_ok=True)

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
random.shuffle(combined)
images, orig_masks, pred_masks = zip(*combined)
images, orig_masks, pred_masks = list(images), list(orig_masks), list(pred_masks)


def find_start_index(files, number):
    for i, f in enumerate(files):
        if re.search(rf"{number}", f):
            return i
    return 0


class LRUCache:
    """Simple LRU cache for images with memory management"""

    def __init__(self, maxsize=20):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                # Remove least recently used
                oldest = self.cache.popitem(last=False)
                del oldest  # Help garbage collection
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        gc.collect()


class ImageMaskViewerOptimized:
    def __init__(self, images, orig_masks, pred_masks, image_dir, orig_mask_dir, pred_mask_dir,
                 start_index=0, cache_size=20, prefetch_count=3):
        self.images = images
        self.orig_masks = orig_masks
        self.pred_masks = pred_masks
        self.image_dir = image_dir
        self.orig_mask_dir = orig_mask_dir
        self.pred_mask_dir = pred_mask_dir
        self.index = start_index
        self.comment = ""

        # Caching and prefetching
        self.cache = LRUCache(cache_size)
        self.prefetch_count = prefetch_count
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_queue = queue.Queue()

        # Create 4 columns: Image | Original Mask | Predicted Mask | Overlay(pred)
        self.fig, self.axs = plt.subplots(1, 4, figsize=(20, 5))
        plt.subplots_adjust(bottom=0.25)

        # Buttons
        axprev = plt.axes([0.15, 0.15, 0.1, 0.07])
        axnext = plt.axes([0.30, 0.15, 0.1, 0.07])
        axaccept = plt.axes([0.55, 0.15, 0.12, 0.07])
        axrework = plt.axes([0.70, 0.15, 0.12, 0.07])
        axbox = plt.axes([0.15, 0.05, 0.67, 0.05])  # text box

        self.bprev = Button(axprev, 'Previous')
        self.bnext = Button(axnext, 'Next')
        self.baccept = Button(axaccept, 'Accept')
        self.brework = Button(axrework, 'Rework')
        self.textbox = TextBox(axbox, "Comment: ")

        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)
        self.baccept.on_clicked(self.accept)
        self.brework.on_clicked(self.rework)
        self.textbox.on_submit(self.update_text)

        # Load initial images and start prefetching
        self._prefetch_around_index(self.index)
        self.update_display()
        plt.show()

    def load_images_sync(self, idx):
        """Synchronously load images for given index"""
        if idx < 0 or idx >= len(self.images):
            return None

        # Check cache first
        cached = self.cache.get(idx)
        if cached is not None:
            return cached

        img_path = os.path.join(self.image_dir, self.images[idx])
        orig_mask_path = os.path.join(self.orig_mask_dir, self.orig_masks[idx])
        pred_mask_path = os.path.join(self.pred_mask_dir, self.pred_masks[idx])

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load original mask
        orig_mask = cv2.imread(orig_mask_path, cv2.IMREAD_UNCHANGED)
        if len(orig_mask.shape) == 2:
            orig_mask_colored = cv2.applyColorMap(orig_mask, cv2.COLORMAP_JET)
        else:
            orig_mask_colored = cv2.cvtColor(orig_mask, cv2.COLOR_BGR2RGB)

        # Load predicted mask
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
        if len(pred_mask.shape) == 2:
            pred_mask_colored = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
        else:
            pred_mask_colored = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB)

        # Create overlay using predicted mask
        overlay = cv2.addWeighted(img, 0.5, pred_mask_colored, 0.5, 0)

        result = (img, orig_mask_colored, pred_mask_colored, overlay)

        # Cache the result
        self.cache.put(idx, result)

        return result

    def _prefetch_around_index(self, center_idx):
        """Prefetch images around the current index"""
        indices_to_prefetch = []

        # Prefetch next few images
        for i in range(1, self.prefetch_count + 1):
            next_idx = center_idx + i
            if next_idx < len(self.images) and self.cache.get(next_idx) is None:
                indices_to_prefetch.append(next_idx)

        # Prefetch previous few images
        for i in range(1, self.prefetch_count + 1):
            prev_idx = center_idx - i
            if prev_idx >= 0 and self.cache.get(prev_idx) is None:
                indices_to_prefetch.append(prev_idx)

        # Submit prefetch tasks
        for idx in indices_to_prefetch:
            self.executor.submit(self._prefetch_single, idx)

    def _prefetch_single(self, idx):
        """Prefetch a single image in background"""
        try:
            self.load_images_sync(idx)
        except Exception as e:
            print(f"Prefetch failed for index {idx}: {e}")

    def update_display(self):
        if not self.images:
            self.fig.suptitle("No more images left!", fontsize=14, color="red")
            for ax in self.axs:
                ax.clear()
                ax.axis("off")
            self.fig.canvas.draw()
            return

        # Load current images (from cache if available)
        result = self.load_images_sync(self.index)
        if result is None:
            return

        img, orig_mask, pred_mask, overlay = result

        # Clear and display
        for ax in self.axs:
            ax.clear()
            ax.axis("off")

        self.axs[0].imshow(img)
        self.axs[0].set_title("Image")

        self.axs[1].imshow(orig_mask)
        self.axs[1].set_title("Original Mask")

        self.axs[2].imshow(pred_mask)
        self.axs[2].set_title("Predicted Mask")

        self.axs[3].imshow(overlay)
        self.axs[3].set_title("Overlay (Pred)")

        self.fig.suptitle(f"[{self.index + 1}/{len(self.images)}] {self.images[self.index]}")
        self.fig.canvas.draw()

        # Trigger prefetching for nearby images
        self._prefetch_around_index(self.index)

    def update_text(self, text):
        self.comment = text

    def move_files(self, dest_img_dir, dest_mask_dir, log_comment=False):
        img_name = self.images[self.index]
        pred_mask_name = self.pred_masks[self.index]
        orig_mask_name = self.orig_masks[self.index]

        # Move image and predicted mask (the ones being reviewed)
        src_img = os.path.join(self.image_dir, img_name)
        src_mask = os.path.join(self.pred_mask_dir, pred_mask_name)

        dst_img = os.path.join(dest_img_dir, img_name)
        dst_mask = os.path.join(dest_mask_dir, pred_mask_name)

        shutil.move(src_img, dst_img)
        shutil.move(src_mask, dst_mask)

        # Log the action
        comment = self.comment + ("Rejected" if log_comment else "Accepted")
        with open(csv_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([img_name, pred_mask_name, comment])

        print(f"Moved {img_name} and {pred_mask_name} -> {dest_img_dir}, {dest_mask_dir}")
        print(f"Logged: {comment}")

        # Remove from cache and lists
        if self.index in self.cache.cache:
            del self.cache.cache[self.index]

        del self.images[self.index]
        del self.orig_masks[self.index]
        del self.pred_masks[self.index]

        # Adjust cache indices (shift down indices > current)
        new_cache = OrderedDict()
        for idx, data in self.cache.cache.items():
            if idx < self.index:
                new_cache[idx] = data
            elif idx > self.index:
                new_cache[idx - 1] = data
        self.cache.cache = new_cache

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

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'cache'):
            self.cache.clear()


# --- Find start index ---
start_index = find_start_index(images, start_number)

# --- Run Optimized Viewer ---
viewer = ImageMaskViewerOptimized(
    images, orig_masks, pred_masks, image_dir, orig_mask_dir, pred_mask_dir,
    start_index=start_index,
    cache_size=15,  # Cache 15 image sets (adjust based on RAM)
    prefetch_count=3  # Prefetch 3 images ahead/behind
)