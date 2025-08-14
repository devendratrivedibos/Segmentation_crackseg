import os
import cv2

# Define your folders
main_image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Devendra"      # Folder with many main images
mask_image_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Devendra\Masks"      # Folder with 4 mask images
output_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Devendra\overlay"       # Output folder

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

mask_files = [f for f in os.listdir(mask_image_folder) if f.endswith('.png')]

# Overlay only if mask filename matches main image filename
for mask_file in mask_files:
    main_img_path = os.path.join(main_image_folder, mask_file)
    mask_img_path = os.path.join(mask_image_folder, mask_file)
    if os.path.exists(main_img_path):
        main_img = cv2.imread(main_img_path, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(main_img.shape)
        mask_img = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
        print(mask_img.shape)

        main_img = cv2.imread(main_img_path)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        print(main_img.shape)
        # Load mask in color and convert to RGB
        # mask_rgb = cv2.imread(self.masks_path[idx], cv2.IMREAD_COLOR)
        # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        if main_img is None or mask_img is None:
            print(f"Failed to load images for {mask_file}")
            continue

        # Resize mask to main image size if necessary
        if main_img.shape[:2] != mask_img.shape[:2]:
            mask_img = cv2.resize(mask_img, (main_img.shape[1], main_img.shape[0]))

        # Blend images (alpha = 0.5 for each, change as needed)
        alpha = 0.5
        blended = cv2.addWeighted(main_img, 1 - alpha, mask_img, alpha, 0)

        save_path = os.path.join(output_folder, mask_file)
        cv2.imwrite(save_path, blended)
        print(f"Saved overlay for {mask_file} at {save_path}")
    else:
        print(f"Main image {mask_file} not found, skipping")

print("Done overlaying mask images using OpenCV.")
