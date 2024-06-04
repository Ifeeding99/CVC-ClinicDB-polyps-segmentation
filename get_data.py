import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
    n_pixels = 0

    for image in os.listdir(img_path):
        im = tifffile.imread(os.path.join(img_path,image))
        h = im.shape[0]
        w = im.shape[1]
        n_pixels += h*w
        im = einops.rearrange(im, 'H W C -> C (H W)')
        mean += im.sum(axis=1) # summing along height and width, output will have shape [3]

    mean /= n_pixels
    for image in os.listdir(img_path):
        im = tifffile.imread(os.path.join(img_path,image))
        im = einops.rearrange(im, 'H W C -> (H W) C')
        std_matrix = (im - mean)**2
        std += std_matrix.sum(axis=0)

    std = np.sqrt(std/n_pixels)
    return mean, std

m,s = calculate_mean_and_std(local_images_path)

t = A.Compose([
    A.Resize(img_size,img_size),
    A.Rotate(
        limit=(-180, 180),
        interpolation=1,
        border_mode=4,
        value=None,
        mask_value=None,
        rotate_method="largest_box",
        crop_border=False,
        always_apply=False,
        p=0.5
    ),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=1,
                       sigma=50,
                       alpha_affine=50),
    A.MotionBlur(blur_limit=11),
    A.Normalize(mean=(102.20527125, 68.70382564, 46.94900982),
                std=(76.10290669, 52.26579002, 35.61231149)), # calculated using the function written above
    ToTensorV2(transpose_mask=True) # to put the mask too in C H W format
])

class PolypsSegmentationDataset(Dataset):
    def __init__(self, images_path = local_images_path, masks_path = local_masks_path, transforms=t):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transforms = transforms
        self.images = os.listdir(self.images_path)
        self.masks = os.listdir(self.masks_path)
        self.l = len(self.images)

    def __getitem__(self, item):
        # the images and the corresponding masks have the same name in the images folder and in the masks folder
        im = tifffile.imread(os.path.join(self.images_path, self.images[item]))
        mask_index = self.masks.index(self.images[item])
        m = tifffile.imread(os.path.join(self.masks_path, self.masks[mask_index]))
        m = np.expand_dims(m,axis=2) # I need the mask to have shape H W 1
        m = m / 255 # to scale between 0 and 1 the masks

        if self.transforms:
            # the output of tiffile.imread() is already a numpy array
            augmented = self.transforms(image=im, mask=m)
            im = augmented['image']
            m = augmented['mask']
        return im,m

    def __len__(self):
        return self.l

def create_dataset(img_path:str, mask_path:str) -> Dataset:
    '''
    This function is used to create the dataset with the transformations used
    Parameters
    ----------
    img_path: str path of the folder containing the images
    mask_path: str path of the folder containing the masks

    Returns: Dataset a pytorch dataset containing images, masks and transforms used
    -------

    '''
    dataset = PolypsSegmentationDataset(img_path,mask_path,transforms=t)
    return dataset
