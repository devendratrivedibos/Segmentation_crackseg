import cv2
import os
from pathlib import Path

# Input and output directories
input_dir = Path(r"C:\Users\Devendra Trivedi\DevendraData\BOS\Crackss\cracks\crack_filter_data_1\IMAGES")
output_dir = Path(r"C:\Users\Devendra Trivedi\DevendraData\BOS\Crackss\cracks\crack_filter_data_1\IMAGES_1024_1024")
output_dir.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# Resize and save
for img_path in input_dir.glob("*"):
    if img_path.suffix.lower() in image_extensions:
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[SKIP] Failed to load image: {img_path}")
            continue

        resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), resized)

        print(f"[OK] Saved resized image to: {save_path}")
