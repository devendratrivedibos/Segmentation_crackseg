import os

def clean_unmatched(img_dir, mask_dir):
    # Collect base names (without extension)
    img_names = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    mask_names = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.lower().endswith(".png")}

    # Find unmatched files
    imgs_to_delete = img_names - mask_names
    masks_to_delete = mask_names - img_names

    # Delete unmatched images
    for f in os.listdir(img_dir):
        base, ext = os.path.splitext(f)
        if base in imgs_to_delete and ext.lower() in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(img_dir, f)
            os.remove(path)
            print(f"🗑 Deleted unmatched image: {path}")

    # Delete unmatched masks
    for f in os.listdir(mask_dir):
        base, ext = os.path.splitext(f)
        if base in masks_to_delete and ext.lower() == ".png":
            path = os.path.join(mask_dir, f)
            os.remove(path)
            print(f"🗑 Deleted unmatched mask: {path}")

    print("✅ Cleanup complete.")


# Example usage
clean_unmatched(

    r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\AnnotationImages",   # your image folder
    r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_CONCRETE\AnnotationMasks"
)
