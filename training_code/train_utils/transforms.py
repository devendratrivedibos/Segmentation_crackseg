import numpy as np
import random
import cv2
import torch

def pad_if_smaller(img, size, fill=0):
    h, w = img.shape[:2]
    pad_h = max(size - h, 0)
    pad_w = max(size - w, 0)
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize(object):
    def __init__(self, img_size):
        self.img_size = img_size  # int or (h, w)
    def __call__(self, image, target):
        if isinstance(self.img_size, int):
            h, w = image.shape[:2]
            scale = self.img_size / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_h, new_w = self.img_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size if max_size is not None else min_size
    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        h, w = image.shape[:2]
        scale = size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 0)
            target = cv2.flip(target, 0)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        h, w = image.shape[:2]
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        image = image[top:top+self.size, left:left+self.size]
        target = target[top:top+self.size, left:left+self.size]
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        h, w = image.shape[:2]
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        image = image[top:top+self.size, left:left+self.size]
        target = target[top:top+self.size, left:left+self.size]
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = image.astype(np.float32) / 255.0
        if image.ndim == 3:
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        else:
            image = torch.from_numpy(image).unsqueeze(0)  # Grayscale
        target = torch.from_numpy(target.astype(np.int64))
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, target):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image, target
