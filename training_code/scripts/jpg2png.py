import os
from PIL import Image

# Folder containing your JPG mask images
jpg_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4040"
# Folder to save PNG images (can be same as jpg_folder if you want)
png_folder = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4040"
os.makedirs(png_folder, exist_ok=True)

# List all JPG files in the folder
jpg_files = [f for f in os.listdir(jpg_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]

for jpg_file in jpg_files:
    jpg_path = os.path.join(jpg_folder, jpg_file)
    # Open the JPG image
    with Image.open(jpg_path) as img:
        # Convert and save as PNG with same file name but .png extension
        png_file = os.path.splitext(jpg_file)[0] + '.png'
        png_path = os.path.join(png_folder, png_file)
        img.save(png_path, 'PNG')
        print(f"Converted {jpg_file} to {png_file}")

print("All JPG images have been converted to PNG format.")
