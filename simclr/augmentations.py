
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F

from skimage.color import rgb2hed
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    """Borrowed from MoCo implementation"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class FixedRandomRotation:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = F.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)


class AlbumentationsTransform:
    """Wrapper for Albumnetation transforms"""
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        aug_img = self.aug(image=np.array(img))['image']
        return aug_img



def torchvision_transforms(eval=False, aug=None):

    trans = []

    if aug["resize"]:
       trans.append(transforms.Resize(aug["resize"]))

    if aug["randcrop"] and aug["scale"] and not eval:
        trans.append(transforms.RandomResizedCrop(aug["randcrop"], scale=aug["scale"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip(p=0.5))
        trans.append(transforms.RandomVerticalFlip(p=0.5))

    if aug["jitter_d"] and not eval:
        trans.append(transforms.RandomApply(
            [transforms.ColorJitter(0.8*aug["jitter_d"], 0.8*aug["jitter_d"], 0.8*aug["jitter_d"], 0.2*aug["jitter_d"])],
             p=aug["jitter_p"]))

    if aug["gaussian_blur"] and not eval:
        trans.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=aug["gaussian_blur"]))

    if aug["rotation"] and not eval:
        # rotation_transform = FixedRandomRotation(angles=[0, 90, 180, 270])
        trans.append(FixedRandomRotation(angles=[0, 90, 180, 270]))

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    # trans = transforms.Compose(trans)
    return trans

def album_transforms(eval=False, aug=None):
    trans = []

    if aug["resize"]:
       trans.append(A.Resize(aug["resize"], aug["resize"], always_apply=True))

    if aug["randcrop"] and not eval:
        #trans.append(A.PadIfNeeded(min_height=aug["randcrop"], min_width=aug["randcrop"]))
        trans.append(A.RandomResizedCrop(width=aug["randcrop"], height=aug["randcrop"], scale=aug["scale"]))

    if aug["randcrop"] and eval:
        #trans.append(A.PadIfNeeded(min_height=aug["randcrop"], min_width=aug["randcrop"]))
        trans.append(A.CenterCrop(width=aug["randcrop"], height=aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(A.Flip(p=0.5))
        #trans.append(A.HorizontalFlip(p=0.5))

    if aug["jitter_d"] and not eval:
        trans.append(A.ColorJitter(0.8*aug["jitter_d"], 0.8*aug["jitter_d"], 0.8*aug["jitter_d"], 0.2*aug["jitter_d"],
                                   p=aug["jitter_p"]))

    if aug["gaussian_blur"] and not eval:
        trans.append(A.GaussianBlur(blur_limit=(3,7), sigma_limit=(0.1, 2), p=aug["gaussian_blur"]))

    if aug["rotation"] and not eval:
        trans.append(A.RandomRotate90(p=0.5))

    if aug["mean"]:
        trans.append(A.Normalize(mean=aug["mean"], std=aug["std"]))

    # Pathology specific augmentation
    if aug["grid_distort"] and not eval:
        trans.append(A.GridDistortion(num_steps=9, distort_limit=0.2, interpolation=1, border_mode=2, p=aug["grid_distort"]))
    if aug["contrast"] and not eval:
        trans.append(A.RandomContrast(limit=aug["contrast"], p=aug["contrast_p"]))
    if aug["grid_shuffle"] and not eval:
        trans.append(A.RandomGridShuffle(grid=(3, 3), p=aug["grid_shuffle"]))

    trans.append(ToTensorV2())

    return trans

def get_rgb_transforms(opt, eval=False):
    aug = {
        "resize": None,
        "randcrop": opt.image_size,
        "scale": opt.scale,
        "flip": True,
        "jitter_d": opt.rgb_jitter_d,
        "jitter_p": opt.rgb_jitter_p,
        "grayscale": False,
        "gaussian_blur": opt.rgb_gaussian_blur_p,
        "rotation": True,
        "contrast": opt.rgb_contrast,
        "contrast_p": opt.rgb_contrast_p,
        "grid_distort": opt.rgb_grid_distort_p,
        "grid_shuffle": opt.rgb_grid_shuffle_p,
        "mean": [0.4914, 0.4822, 0.4465],  # values for train+unsupervised combined
        "std": [0.2023, 0.1994, 0.2010],
        "bw_mean": [0.4120],  # values for train+unsupervised combined
        "bw_std": [0.2570],
    }
    if opt.use_album:
        return transforms.Compose([AlbumentationsTransform(A.Compose(album_transforms(eval=eval, aug=aug)))])

    return transforms.Compose(torchvision_transforms(eval=eval, aug=aug))


def get_transforms(opt, eval=False):
    return get_rgb_transforms(opt, eval)


