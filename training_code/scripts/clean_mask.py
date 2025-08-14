import os
import cv2
import numpy as np

# Your color map (RGB → ID)
COLOR_MAP = {
    (0, 0, 0): 0,           # Background
    (255, 0, 0): 1,         # Alligator Crack
    (0, 0, 255): 2,         # Transverse Crack
    (0, 255, 0): 3,         # Longitudinal Crack
    (255, 0, 255): 4,       # Multiple Crack
    (255, 204, 0): 5,       # Joint Seal
    (0, 42, 255): 6         # Pothole
}

# Convert keys to a NumPy array for efficient comparison
known_colors = np.array(list(COLOR_MAP.keys()), dtype=np.uint8)  # shape: (N, 3)

# Input and output folders
input_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\segmentation_dataset_08_aug\Annotation 8 august 2025\Tanishq\Masks"
output_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\segmentation_dataset_08_aug\Annotation 8 august 2025\Tanishq\Masks"
os.makedirs(output_folder, exist_ok=True)
# Process all images
for fname in os.listdir(input_folder):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

        fpath = os.path.join(input_folder, fname)

        image_bgr = cv2.imread(fpath)
        if image_bgr is None:
            continue

        # Convert to RGB for matching
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Flatten to (H*W, 3)
        flat = image_rgb.reshape(-1, 3)

        # Create a boolean mask: True if pixel matches any known color
        match = np.any(np.all(flat[:, None] == known_colors[None, :], axis=-1), axis=1)

        # Replace unmatched pixels with (0, 0, 0)
        flat[~match] = (0, 0, 0)

        # Reshape back to image
        cleaned_rgb = flat.reshape(image_rgb.shape)

        # Convert back to BGR for saving
        cleaned_bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)

        # Save to output folder
        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, cleaned_bgr)

print("✅ All masks cleaned and saved to:", output_folder)
