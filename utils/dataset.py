'''
A wrapper for naive dataset.
'''

import os
from torchvision import datasets, transforms

import torch
from torch.utils.data.dataset import Dataset

from utils import registry
from module import pca


@registry.register('Dataset', 'Train')
class TrainDataset(Dataset):
    def __init__(self, n_top=7):
        # @TODO
        original_path = ""  # where to load original images
        beautify_path = ""  # where to load beautified imgs
        self.image_x = ??  # load beautified images: N x H x W x 3
        image_y = ??       # load ground truth imgs: N x H x W x 3
        self.image_e = (image_y - image_x).view(N, -1)  # N x (H*W*3), every entry is a vector
        # TODO:
        eigenvectors, eigenvalues = PCA(image_e)  # orrdered by the value of eigenvelues
        self.eigenvectors = eigenvectors[:n_top]
        self.eigenvalues = eigenvectors[:n_top+1]

    def get_(self):
        return self.eigenvalues

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        x = self.image_x[idx]  # H x W x 3
        e = self.image_e[idx]  # (H*W*3)
        # @TODO
        components = [projection(e, v) for v in self.eigenvectors]
        components += [e - torch.sum(componnts)]
        y = torch.stack([component/val for component, val in zip(components, self.eigenvalues)]).view(-1, H, W, 3)
        return [x, y]


@registry.register('Dataset', 'Valid')
class ValidDataset(Dataset):
    def __init__(self, n_top=7):
        # @TODO
        original_path = ""  # where to load original images
        beautify_path = ""  # where to load beautified imgs
        self.image_x = ??  # load beautified images: N x H x W x 3
        self.image_y = ??  # load ground truth imgs: N x H x W x 3

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        x = self.image_x[idx]
        y = self.image_y[idx]
        return [x, y]


