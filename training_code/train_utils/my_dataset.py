import os
import pdb
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# COLOR_MAP = {
#     (0, 0, 0): 0,  # Black   Background
#     (255, 0, 0): 1,  # Red      Alligator
#     (0, 0, 255): 2,  # Blue     Transverse Crack
#     (0, 255, 0): 3,  # Green    Longitudinal Crack
#     (255, 0, 255): 4,  # Magenta  Multiple Crack
#     (255, 204, 0): 5,  # Yellow   Joint Seal
#     (0, 42, 255): 6,  # Orange   Pothole
# }

COLOR_MAP = {
    (0, 0, 0): 0,  # Black   - Background     1139
    (255, 0, 0): 1,  # Red     - Alligator      700
    (0, 0, 255): 2,  # Blue    - Transverse Crack    90
    (0, 255, 0): 3,  # Green   - Longitudinal Crack    522
    (139, 69, 19): 4,  # Brown    -  POTHOLE
    (255, 165, 0): 5,  # Orange   - PATCHES
    (128, 0, 128): 6,  # Purple   - punchout
    (0, 255, 255): 7,  # Cyan     -spalling
    (0, 128, 0): 8,  # Dark Green   - COrner Break
    (255, 100, 203): 9,  # Light Pink    - SEALED JOINT - T
    (199, 21, 133): 10,  # Dark Pink  - SEALED JOINT - L
    (255, 215, 0): 11,  # Gold    CRACKING
    (255, 255, 255): 12,  # WHITE  UNCLASSIFIED
}


class CrackDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(CrackDataset, self).__init__()
        data_root = root
        flag = "TRAIN" if train else "VAL"

        data_root = os.path.join(data_root, flag)
        assert os.path.exists(data_root), "path '{}' does not exist.".format(data_root)
        # data_root = root
        imgs_root = os.path.join(data_root, "IMAGES")
        masks_root = os.path.join(data_root, "MASKS")

        self.images_list = os.listdir(imgs_root)
        valid_exts = ['.png', '.jpg', '.jpeg']
        self.images_list = [f for f in os.listdir(imgs_root) if Path(f).suffix.lower() in valid_exts]

        self.images_path = [os.path.join(imgs_root, i) for i in self.images_list]

        # self.masks_path = [os.path.join(masks_root, i) for i in self.images_list]  # same_name

        # Build a lookup dictionary for masks (key = base name without extension)

        self.masks_path = [os.path.join(masks_root, os.path.splitext(i)[0] + '.png')
                            for i in self.images_list]
        print("LEN IMAGES:", len(self.images_path))
        print("LEN MASKS:", len(self.masks_path))
        assert (len(self.images_path) == len(self.masks_path))

        self.transforms = transforms

    def __getitem__(self, idx):
        # Load image using OpenCV and convert to RGB
        img = cv2.imread(self.images_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        # Load mask in color and convert to RGB
        mask_rgb = cv2.imread(self.masks_path[idx], cv2.IMREAD_COLOR)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        # mask_rgb = cv2.resize(mask_rgb, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        # Convert RGB mask to class ID map
        mask = self.rgb_to_class_id(mask_rgb, COLOR_MAP)
        if self.transforms is not None:
            # img, mask = self.transforms(img, mask)
            result = self.transforms(img, mask)
            img, mask = result["image"], result["mask"]
        mask = mask.long()
        return img, mask

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    def rgb_to_class_id(self, mask_rgb, color2id):
        # Create a blank class map
        class_map = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        for color, class_id in color2id.items():
            # Create a boolean mask where all pixels match the color
            match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
            class_map[match] = class_id
        return class_map


class SegmentationPresetTrain:
    def __init__(self, img_size, mean, std):
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.CLAHE(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img, target):
        result = self.transforms(image=img, mask=target)
        return result
        # return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, img_size, mean, std):
        self.transforms = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img, target):
        result = self.transforms(image=img, mask=target)
        return result


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
