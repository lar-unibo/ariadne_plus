from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms as TF 
import cv2


class BasicDataset(Dataset):
    def __init__(self, folder, transform):

        self.imgs_dir = f"{folder}/imgs/"
        self.masks_dir = f"{folder}/masks/"
        self.transform = transform

        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255    

        return img    

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = f"{self.masks_dir}{idx}.png"
        img_file = f"{self.imgs_dir}{idx}.png"

        img = np.array(Image.open(img_file))
        mask = np.array(Image.open(mask_file))

        data = {"image": img, "mask": mask}
        augmented = self.transform(**data)
        img, mask = augmented["image"], augmented["mask"] 

        # HWC to CHW
        img = self.pre_process(img)
        mask = self.pre_process(mask)

        return {'image': torch.from_numpy(img).type(torch.FloatTensor), 
                'mask': torch.from_numpy(mask).type(torch.FloatTensor)}
