import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from PIL import Image
import os
import einops
from typing import Tuple
import tifffile

# this scripts is used to define a Dataset class which retrieves the images and the masks
# the augmentations are also defined here

local_images_path = 'C:/Users/flavi/Downloads/CVC-ClinicDB/Original'
local_masks_path = 'C:/Users/flavi/Downloads/CVC-ClinicDB/Ground_Truth'
img_size = 256

def calculate_mean_and_std(img_path: str) -> Tuple[float, float]:
    '''
    This functions is used to calculate mean and standard deviation of the images in the dataset
    Parameters
    ----------
    img_path: local path to images

    Returns Tuple[float,float], the first number will be the calculated mean and the second the calculated std
    -------
    '''

    mean = np.zeros(3)
    std = np.zeros(3)
    n_images = 0

    for image in os.listdir(img_path):
        im = tifffile.imread(os.path.join(img_path,image))
        h = im.shape[0]
        w = im.shape[1]
        n_pixels = h*w
        im = einops.rearrange(im, 'H W C -> (H W) C')
        a = im.sum(axis=0)
        mean += im.sum(axis=0) # summing along height and width, output will have shape [3]
        std += np.sum(im**2, axis=0)
        n_images += 1

    mean /=( n_images * n_pixels)
    std = np.sqrt(std/( n_images * n_pixels))
    return mean, std



class PolypsSegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, transforms=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transforms = transforms
        self.images = os.listdir(self.images_path)
        self.masks = os.listdir(self.masks_path)
        self.l = len(images)

    def __getitem__(self, item):
        # the images and the corresponding masks have the same name in the images folder and in the masks folder
        im = Image.open(os.path.join(self.images_path, self.images[item])).convert('RGB') # to ensure they are in RGB format
        mask_index = self.masks.index(self.images[item]).convert('L') # to ensure the masks are grayscale
        m = Image.open(os.path.join(self.masks_path, self.masks[mask_index]))

        if self.transforms:
            im = np.array(im)
            m = np.array(m)
            augmented = self.transforms(image=im, mask=m)
            img = augmented['image']
            m = augmented['mask']
            return img,m

m,s = calculate_mean_and_std(local_images_path)
print(m)
print(s)
