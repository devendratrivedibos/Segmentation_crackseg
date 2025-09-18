import os
import cv2
import numpy as np

# --- Define Folders ---
folders = [
    # r'Y:\NSV_DATA\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-4\AnnotationMasks',
]

# Color replacement (BGR because OpenCV loads in BGR)
old_color = (255, 255, 0)  # (R=255,G=192,B=203) → BGR order
# old_color = (112, 102, 255)  # (R=255,G=100,B=203) → BGR order
new_color = (0, 255, 255)
# Loop through folders
for folder in folders:
    for f in os.listdir(folder):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        path = os.path.join(folder, f)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            print(f"⚠️ Could not read: {path}")
            continue
        # Create mask where pixels match the old color
        mask = np.all(img == old_color, axis=-1)
        if np.any(mask):
            img[mask] = new_color
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img)  # overwrite the file
            print(f"✅ Fixed: {path}")
        else:
            print(f"Skipped (no target color): {path}")

img_paths = [
# r"Y:\NSV_DATA\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33\AnnotationMasksNIGHT\DAGMAGPUR-LALGANJ_2024-10-04_16-13-33_SECTION-4_IMG_0000173.png"
]
old_color = (255, 255, 255)
new_color = (0, 255, 255)
for img_path in img_paths:

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        print(f"⚠️ Could not read: {img_path}")
    # Create mask where pixels match the old color
    mask = np.all(img == old_color, axis=-1)
    if np.any(mask):
        img[mask] = new_color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img)  # overwrite the file
        print(f"✅ Fixed: {img_path}")
    else:
        print(f"Skipped (no target color): {img_path}")


img = cv2.imread(r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\ACCEPTED_MASKS\HAZARIBAGH-RANCHI_2024-10-07_11-25-27_SECTION-2_IMG_0002097.png ")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = np.zeros_like(img)
result[np.all(img == (0,255,0), axis=-1)] = (0,255,0)
img = cv2.imwrite(r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\ACCEPTED_MASKS\HAZARIBAGH-RANCHI_2024-10-07_11-25-27_SECTION-2_IMG_0002097.png", result)