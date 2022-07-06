import torchvision
import torch
import torchvision.transforms as tfs
import models
import os
import util
from torch.utils.data import DataLoader
from augment import Augment, Cutout
from randaugment import RandAugmentMC


# from prefetch_generator import BackgroundGenerator


# class DataLoaderX(DataLoader):

#    def __iter__(self):
#        return BackgroundGenerator(super().__iter__())

class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""

    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


def get_aug_dataloader(image_dir, is_validation=False,
                       batch_size=128,
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
            tfs.Resize(256),
            tfs.CenterCrop(224),
            tfs.ToTensor(),
            normalize
        ])
    elif augs == 1:
        _transforms = tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(224),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            normalize
        ])
    elif augs == 2:
        _transforms = tfs.Compose([
            tfs.Resize(256),
            tfs.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            tfs.ColorJitter(0.5, 0.5, 0.5, 0.5),
            tfs.RandomGrayscale(p=0.5),
            tfs.RandomHorizontalFlip(),
            tfs.RandomRotation(degrees=(-25, 25)),
            tfs.ToTensor(),
            normalize
        ])
    elif augs == 3:
        _transforms = tfs.Compose([
            tfs.Resize(256),
            tfs.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            tfs.RandomHorizontalFlip(),
            RandAugmentMC(n=6, m=6),
            tfs.ToTensor(),
            normalize
        ])
    elif augs == 4:
        _transforms = tfs.Compose([
           tfs.Resize(256),
           tfs.RandomResizedCrop(size=224, scale=(0.2, 1.)),
           tfs.RandomHorizontalFlip(),
           Augment(10),
           tfs.ToTensor(),
           normalize,
           Cutout(
               n_holes=4,
               length=32,
               random=True)])
    if is_validation:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/val', _transforms))
    else:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/train', _transforms))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return loader
