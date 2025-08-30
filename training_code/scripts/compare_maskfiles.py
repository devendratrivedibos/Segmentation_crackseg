import os

# Paths
folder1 = r"X:\DataSet\DATASET_IMAGES_"
folder2 = r"X:\DataSet\DATASET_MASKS_"

# Get filenames without path
files1 = {f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
files2 = {f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}
# Get base filenames (without extension)
files1 = {os.path.splitext(f)[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
files2 = {os.path.splitext(f)[0] for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}

# Files in folder1 but not in folder2
diff_files = sorted(files1 - files2)

print(f"Total unique files in folder1 not in folder2: {len(diff_files)}")
for f in diff_files:
    print(f)