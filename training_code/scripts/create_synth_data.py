import os
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


# ============================================================
# COLOR MAP
# ============================================================
COLOR_MAP = {
    (0, 0, 0): (0, "Background"),
    (255, 0, 0): (1, "Alligator"),
    (0, 0, 255): (2, "Transverse Crack"),
    (0, 255, 0): (3, "Longitudinal Crack"),
    (139, 69, 19): (4, "Pothole"),
    (255, 165, 0): (5, "Patches"),
    (255, 0, 255): (6, "Multiple Crack"),
    (0, 255, 255): (7, "Spalling"),
    (0, 128, 0): (8, "Corner Break"),
    (255, 100, 203): (9, "Sealed Joint Transverse"),
    (199, 21, 133): (10, "Sealed Joint Longitudinal"),
    (128, 0, 128): (11, "Punchout"),
    (112, 102, 255): (12, "Popout"),
    (255, 255, 255): (13, "Unclassified"),
    (255, 215, 0): (14, "Cracking"),
}

# ============================================================
# CLASSES TO AUGMENT
# ============================================================

TARGET_CLASSES = [
    "Transverse Crack",
    "Longitudinal Crack",
]

TARGET_COLORS = []
for rgb_color, (_, cls_name) in COLOR_MAP.items():
    if cls_name in TARGET_CLASSES:
        bgr_color = tuple(reversed(rgb_color))
        TARGET_COLORS.append(bgr_color)

print("Target Classes:", TARGET_CLASSES)
print("Target Colors :", TARGET_COLORS)

# ============================================================
# CONFIG
# ============================================================

CANVAS_HEIGHT = 1024
CANVAS_WIDTH = 419

OBJECTS_PER_IMAGE = 50
NUM_SYNTHETIC_IMAGES = 200
INPUT_IMG_DIR = r"Z:\Devendra\ASPHALT\SPLIT\TRAIN\IMAGES"
INPUT_MASK_DIR = r"Z:\Devendra\ASPHALT\SPLIT\TRAIN\MASKS"
BACKGROUND_IMG_DIR = INPUT_IMG_DIR
BACKGROUND_MASK_DIR = INPUT_MASK_DIR

OUT_IMG_DIR = r"Z:\Devendra\ASPHALT\SYNTHETIC\IMAGES"
OUT_MASK_DIR = r"Z:\Devendra\ASPHALT\SYNTHETIC\MASKS"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


BG_IMAGES = []

for ext in IMAGE_EXTENSIONS:
    BG_IMAGES.extend(glob(os.path.join(BACKGROUND_IMG_DIR, "*" + ext)))

print(f"Found {len(BG_IMAGES)} background images")


def find_image(base_name, image_dir):
    """
    Finds image irrespective of extension.
    """
    for ext in IMAGE_EXTENSIONS:
        img_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path

    return None


def get_target_images(mask_dir):
    selected_images = []
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
    for mask_file in tqdm(
            mask_files,
            desc="Finding target images",
            unit="mask"):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path)
        if mask is None:
            continue
        found = False
        for color in TARGET_COLORS:
            pixels = np.all(mask == color, axis=2)
            if np.any(pixels):
                found = True
                break

        if found:
            selected_images.append(os.path.splitext(mask_file)[0])
    return selected_images


def extract_objects_from_image(img_path, mask_path):
    objects = []
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if img is None or mask is None:
        return objects

    for color in TARGET_COLORS:
        color_mask = cv2.inRange(mask,
            np.array(color, dtype=np.uint8),
            np.array(color, dtype=np.uint8))

        contours, _ = cv2.findContours( color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            obj_img = img[y:y+h, x:x+w].copy()
            obj_mask = color_mask[y:y+h, x:x+w].copy()
            if obj_img.size == 0:
                continue
            objects.append((obj_img, obj_mask, color))
    return objects


def random_transform(obj_img, obj_mask):
    # ============================================================
    # RANDOM AUGMENTATION
    # ============================================================
    h, w = obj_img.shape[:2]
    scale = random.uniform(0.7, 1.5)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    obj_img = cv2.resize(obj_img,(nw, nh),interpolation=cv2.INTER_LINEAR)
    obj_mask = cv2.resize(obj_mask,(nw, nh),interpolation=cv2.INTER_NEAREST)
    angle = random.uniform(-30, 30)
    center = (nw // 2, nh // 2)

    M = cv2.getRotationMatrix2D(center,angle,1.0)
    obj_img = cv2.warpAffine(obj_img,
        M,(nw, nh),flags=cv2.INTER_LINEAR,borderValue=(0, 0, 0))

    obj_mask = cv2.warpAffine(obj_mask,M, (nw, nh),flags=cv2.INTER_NEAREST,borderValue=0)

    if random.random() < 0.5:
        obj_img = cv2.flip(obj_img, 1)
        obj_mask = cv2.flip(obj_mask, 1)

    if random.random() < 0.5:
        obj_img = cv2.flip(obj_img, 0)
        obj_mask = cv2.flip(obj_mask, 0)

    return obj_img, obj_mask


# ============================================================
# CREATE SYNTHETIC SAMPLE
# ============================================================

def create_synthetic_image(
        image_names,
        save_prefix):
    bg_images = []

    for ext in IMAGE_EXTENSIONS:
        bg_images.extend(
            glob(os.path.join(BACKGROUND_IMG_DIR, "*" + ext))
        )

    bg_img_path = random.choice(bg_images)

    bg_file = os.path.basename(bg_img_path)
    bg_img_path = os.path.join(
        BACKGROUND_IMG_DIR,
        bg_file
    )

    bg_mask_path = os.path.join(BACKGROUND_MASK_DIR, os.path.splitext(bg_file)[0] + ".png")

    synth_img = cv2.imread(bg_img_path)

    if synth_img is None:
        return

    synth_mask = cv2.imread(bg_mask_path)

    if synth_mask is None:
        synth_mask = np.zeros_like(synth_img)

    synth_img = cv2.resize(
        synth_img,
        (CANVAS_WIDTH, CANVAS_HEIGHT)
    )

    synth_mask = cv2.resize(
        synth_mask,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )

    # -------------------------------------------------------
    # COLLECT OBJECTS
    # -------------------------------------------------------

    all_objects = []

    selected_imgs = random.sample(
        image_names,
        min(20, len(image_names))
    )

    for img_name in selected_imgs:

        img_path = find_image(
            img_name,
            INPUT_IMG_DIR
        )

        if img_path is None:
            continue

        mask_path = os.path.join(
            INPUT_MASK_DIR,
            img_name + ".png"
        )
        if not os.path.exists(mask_path):
            continue
        objs = extract_objects_from_image(
            img_path,
            mask_path
        )

        all_objects.extend(objs)

    if len(all_objects) == 0:
        return

    chosen_objects = random.sample(
        all_objects,
        min(
            OBJECTS_PER_IMAGE,
            len(all_objects)
        )
    )

    # -------------------------------------------------------
    # PASTE OBJECTS
    # -------------------------------------------------------

    for obj_img, obj_mask, color in chosen_objects:

        obj_img, obj_mask = random_transform(
            obj_img,
            obj_mask
        )

        oh, ow = obj_img.shape[:2]

        if oh >= CANVAS_HEIGHT:
            continue

        if ow >= CANVAS_WIDTH:
            continue

        x_offset = random.randint(
            0,
            CANVAS_WIDTH - ow
        )

        y_offset = random.randint(
            0,
            CANVAS_HEIGHT - oh
        )

        roi_img = synth_img[
                  y_offset:y_offset+oh,
                  x_offset:x_offset+ow
                  ]

        roi_mask = synth_mask[
                   y_offset:y_offset+oh,
                   x_offset:x_offset+ow
                   ]

        mask_bool = obj_mask > 0

        roi_img[mask_bool] = obj_img[mask_bool]
        roi_mask[mask_bool] = color

        synth_img[
        y_offset:y_offset+oh,
        x_offset:x_offset+ow
        ] = roi_img

        synth_mask[
        y_offset:y_offset+oh,
        x_offset:x_offset+ow
        ] = roi_mask

    cv2.imwrite(
        os.path.join(
            OUT_IMG_DIR,
            save_prefix + ".jpg"
        ),
        synth_img
    )

    cv2.imwrite(
        os.path.join(
            OUT_MASK_DIR,
            save_prefix + ".png"
        ),
        synth_mask
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    image_names = get_target_images(
        INPUT_MASK_DIR
    )

    print(
        f"Found {len(image_names)} images "
        f"containing target classes."
    )

    if len(image_names) == 0:
        raise Exception(
            "No images found containing target classes."
        )

    for idx in range(NUM_SYNTHETIC_IMAGES):

        create_synthetic_image(
            image_names,
            f"synthetic_{idx:04d}"
        )

        print(
            f"Generated "
            f"{idx+1}/{NUM_SYNTHETIC_IMAGES}"
        )

    print("Finished.")