import torchvision
import torch
import torchvision.transforms as tfs
import models
import os
import util
from torch.utils.data import DataLoader

import dill as pickle
from ImageFolder_ import ImageFolder


class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        path, data, target = self.dt[index]
        return path, data, target, index

    def __len__(self):
        return len(self.dt)


def get_aug_dataloader(image_dir, is_validation=False,
                       batch_size=256, image_size=256, crop_size=224,
                       mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],
                       num_workers=8,
                       augs=1, shuffle=True):

    # print(image_dir)
    if image_dir is None:
        return None

    #print("imagesize: ", image_size, "cropsize: ", crop_size)
    normalize = tfs.Normalize(mean=mean, std=std)
    if augs == 0:
        _transforms = tfs.Compose([
            tfs.Resize(image_size),
            tfs.CenterCrop(crop_size),
            tfs.ToTensor(),
            normalize
        ])
    elif augs == 1:
        _transforms = tfs.Compose([
            tfs.Resize(image_size),
            tfs.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=mean, std=std),
        ])
    elif augs == 2:
        _transforms = tfs.Compose([
            tfs.RandomResizedCrop(size=96, scale=(0.2, 1.)),
            tfs.RandomGrayscale(p=0.2),
            tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            normalize
        ])

    if is_validation:
        dataset = DataSet(ImageFolder(image_dir + '/val', _transforms))
    else:
        dataset = DataSet(ImageFolder(image_dir + '/train', _transforms))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return loader