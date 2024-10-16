import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import tifffile as tf
from matplotlib import pyplot as plt
import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


class HybridDataset(Dataset):
    def __init__(self, image_dir, BF_mask_dir, HM_mask_dir, matrix_mask_dir, void_mask_dir, potting_mask_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.BF_mask_dir = BF_mask_dir
        self.HM_mask_dir = HM_mask_dir
        self.matrix_mask_dir = matrix_mask_dir
        self.void_mask_dir = void_mask_dir
        self.potting_mask_dir = potting_mask_dir

 
        self.images = sorted(os.listdir(image_dir))
        self.BF_mask = sorted(os.listdir(BF_mask_dir))
        self.HM_mask = sorted(os.listdir(HM_mask_dir))
        self.matrix_mask = sorted(os.listdir(matrix_mask_dir))
        self.void_mask = sorted(os.listdir(void_mask_dir))
        self.potting_mask = sorted(os.listdir(potting_mask_dir))
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        BF_mask_path = os.path.join(self.BF_mask_dir, self.BF_mask[index])
        HM_mask_path = os.path.join(self.HM_mask_dir, self.HM_mask[index])
        matrix_mask_path = os.path.join(self.matrix_mask_dir, self.matrix_mask[index])
        void_mask_path = os.path.join(self.void_mask_dir, self.void_mask[index])
        potting_mask_path = os.path.join(self.potting_mask_dir, self.potting_mask[index])
 
        img = np.array(tf.imread(img_path))
        BF_mask = np.array(tf.imread(BF_mask_path))
        HM_mask = np.array(tf.imread(HM_mask_path))
        matrix_mask = np.array(tf.imread(matrix_mask_path))
        void_mask = np.array(tf.imread(void_mask_path))
        potting_mask = np.array(tf.imread(potting_mask_path))


        masks = np.stack([BF_mask, HM_mask, matrix_mask, void_mask, potting_mask], axis=0)


        mask = np.argmax(masks, axis=0) #class indexed mask


        if mask.ndim == 3:#augmentation input in numpy array image dim [HWC]
            mask = np.transpose(mask, (1, 2, 0))
        else:

            pass
        

        if img.ndim == 3:#augmentation input in numpy array image dim [HWC]
            img = np.transpose(img, (1, 2, 0))
        else:

            pass

        
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask) 
            img = augmentations['image']
            mask = augmentations['mask']
            
            
        return img, mask

def test():
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Transpose(p=0.2),
            A.Rotate(limit=45, p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(var_limit=(50.0, 100.0), p=0.2),
            A.RandomGamma(p=0.4),
            A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),

            ToTensorV2(transpose_mask=True),
        ]
    )

    
    dataset = HybridDataset(
        image_dir = r'path/img',
        BF_mask_dir =  r'path/mask1',
        HM_mask_dir = r'path/mask2',
        matrix_mask_dir = r'path/mask3',
        void_mask_dir = r'path/mask4',
        potting_mask_dir = r'path/mask5',
        transform=train_transform
    )
    
    img, mask = dataset[1]
    print('Dataset length:', len(dataset))
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")


if __name__ == "__main__":
    test()

