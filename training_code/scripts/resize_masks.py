import cv2
import os
target_size = (4183, 10217)  # width, height

mask_dir = r"Y:\NSV_DATA\HAZARIBAGH-RANCHI_2024-10-07_11-25-27\SECTION-2\ACCEPTED_MASKS"
output_dir = r"Y:\NSV_DATA\HAZARIBAGH-RANCHI_2024-10-07_11-25-27\SECTION-2\HIGH_RES_MASKS"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(mask_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        in_path = os.path.join(mask_dir, fname)
        out_path = os.path.join(output_dir, fname)
        mask = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        # Use nearest-neighbor interpolation to avoid altering label values
        resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_path, resized)
print("✅ Resizing complete! Resized masks saved to:", output_dir)


mask_dir = r"Y:\NSV_DATA\HAZARIBAGH-RANCHI_2024-10-07_11-25-27\SECTION-3\ACCEPTED_MASKS"
output_dir = r"Y:\NSV_DATA\HAZARIBAGH-RANCHI_2024-10-07_11-25-27\SECTION-3\HIGH_RES_MASKS"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(mask_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        in_path = os.path.join(mask_dir, fname)
        out_path = os.path.join(output_dir, fname)
        mask = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        # Use nearest-neighbor interpolation to avoid altering label values
        resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_path, resized)
print("✅ Resizing complete! Resized masks saved to:", output_dir)


mask_dir = r"Y:\NSV_DATA\HAZARIBAGH-RANCHI_2024-10-07_11-25-27\SECTION-4\ACCEPTED_MASKS"
output_dir = r"Y:\NSV_DATA\HAZARIBAGH-RANCHI_2024-10-07_11-25-27\SECTION-4\HIGH_RES_MASKS"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(mask_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        in_path = os.path.join(mask_dir, fname)
        out_path = os.path.join(output_dir, fname)
        mask = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        # Use nearest-neighbor interpolation to avoid altering label values
        resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_path, resized)
print("✅ Resizing complete! Resized masks saved to:", output_dir)