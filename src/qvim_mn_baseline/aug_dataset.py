import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class AugDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        resize=(224, 224),
        aug_profile='light',  # 'light' or 'full'
        # individual override probabilities (None means use profile defaults)
        p_flip=None,
        p_jitter=None,
        p_blur=None,
        p_noise=None,
        p_rotate=None,
    ):
        """
        image_paths: list of file paths
        labels: list of labels
        resize: target image size

        aug_profile: selects preset augmentations ('light' or 'full')
        Individual p_* override specific profile defaults if not None.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.resize = resize

        # set default probabilities based on profile
        profiles = {
            'light': dict(
                p_flip=0.5,
                p_jitter=0.3,
                p_blur=0.0,
                p_noise=0.0,
                p_rotate=0.0,
            ),
            'full': dict(
                p_flip=0.5,
                p_jitter=0.5,
                p_blur=0.2,
                p_noise=0.1,
                p_rotate=0.3,
            ),
        }
        if aug_profile not in profiles:
            raise ValueError(f"Unknown aug_profile '{aug_profile}', choose from {list(profiles.keys())}")

        # start with profile defaults
        defaults = profiles[aug_profile]
        # override with provided values if not None
        self.p_flip   = p_flip   if p_flip   is not None else defaults['p_flip']
        self.p_jitter = p_jitter if p_jitter is not None else defaults['p_jitter']
        self.p_blur   = p_blur   if p_blur   is not None else defaults['p_blur']
        self.p_noise  = p_noise  if p_noise  is not None else defaults['p_noise']
        self.p_rotate = p_rotate if p_rotate is not None else defaults['p_rotate']

        # build transform pipeline
        aug_list = []
        # flip
        if self.p_flip > 0:
            aug_list.append(T.RandomHorizontalFlip(p=self.p_flip))
        # color jitter
        if self.p_jitter > 0:
            aug_list.append(T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=self.p_jitter))
        # blur
        if self.p_blur > 0:
            aug_list.append(T.RandomApply([
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            ], p=self.p_blur))
        # noise (via lambda)
        if self.p_noise > 0:
            aug_list.append(T.RandomApply([
                T.Lambda(lambda img: add_gaussian_noise(img, std=0.1))
            ], p=self.p_noise))
        # rotate
        if self.p_rotate > 0:
            aug_list.append(T.RandomRotation(degrees=30))

        # always resize and to-tensor
        base_transforms = [T.Resize(self.resize), T.ToTensor()]

        self.transform = T.Compose(aug_list + base_transforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label


def add_gaussian_noise(img, std=0.1):
    """
    Add Gaussian noise to a PIL image and return a PIL image.
    """
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)
