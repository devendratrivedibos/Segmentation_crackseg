import os
import cv2
import numpy as np

# --- Define Folders ---
folders = [
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-4\Masks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-3\Masks",
    r"X:\THANE-BELAPUR_2025-05-11_07-35-42\SECTION-2\Masks",
]

# Color replacement (BGR because OpenCV loads in BGR)
old_color = (203, 192, 255)   # (R=255,G=192,B=203) → BGR order
new_color = (203, 100, 255)   # (R=255,G=100,B=203) → BGR order

# Loop through folders
for folder in folders:
    for f in os.listdir(folder):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        path = os.path.join(folder, f)
        img = cv2.imread(path)

        if img is None:
            print(f"⚠️ Could not read: {path}")
            continue

        # Create mask where pixels match the old color
        mask = np.all(img == old_color, axis=-1)

        if np.any(mask):
            img[mask] = new_color
            cv2.imwrite(path, img)  # overwrite the file
            print(f"✅ Fixed: {path}")
        else:
            print(f"Skipped (no target color): {path}")
