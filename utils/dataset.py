'''
A wrapper for naive dataset.
'''

import os
from torchvision import datasets, transforms
from skimage import io

import torch
from torch.utils.data.dataset import Dataset

from utils import registry
from module import pca

import numpy as np


patch_m = 11


@registry.register('Dataset', 'PCA')
class PCADataset(Dataset):
    def __init__(self, path, n_top=7):
        # @TODO
        original_path = path + "original/"  # where to load original images
        beautify_path = path + "beautified/"  # where to load beautified imgs
        self.image_x = load_all_images(beautify_path)  # load beautified images: N x H x W x 3
        self.image_y = load_all_images(original_path)      # load ground truth imgs: N x H x W x 3
        self.image_e = get_patches(image_y - image_x)  # N x (H*W*3), every entry is a vector
        # N_total * patch_m x patch_m x 3
        # TODO:
        my_pca = PCA(image_e)
        # get all patch_m*patch_m*channels evecs and evals
        eigenvectors, eigenvalues = my_pca.get_evec_eval() # orrdered by the value of eigenvelues
        self.eigenvectors = eigenvectors[:n_top]
        self.eigenvalues = eigenvectors[:n_top+1]
        print(self.eigenvectors)
        print(self.eigenvalues)
        print(type(eigenvectors), type(eigenvalues))
        np.save("eigenvalue.npy", eigenvalues)
        np.save("eigenvector.npy", eigenvectors)

    def get_(self):
        return self.eigenvalues

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        x = self.image_x[idx]
        y = self.image_y[idx]
        return [x, y]


@registry.register('Dataset', 'Train')
class TrainDataset(Dataset):
    def __init__(self, path, n_top=7):
        # @TODO
        original_path = path + "original/"  # where to load original images
        beautify_path = path + "beautified/"  # where to load beautified imgs
        self.image_x = load_all_images(beautify_path)  # load beautified images: N x H x W x 3
        image_y = load_all_images(original_path)      # load ground truth imgs: N x H x W x 3
        self.image_e = get_patches(image_y - image_x)  # N x (H*W*3), every entry is a vector
        # N_total * patch_m x patch_m x 3
        # TODO:
        my_pca = PCA(image_e)
        # get all patch_m*patch_m*channels evecs and evals
        eigenvectors, eigenvalues = my_pca.get_evec_eval() # orrdered by the value of eigenvelues
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
        original_path = path + "original/"  # where to load original images
        beautify_path = path + "beautified/"  # where to load beautified imgs
        self.image_x = load_all_images(beautify_path)  # load beautified images: N x H x W x 3
        self.image_y = load_all_images(beautify_path)  # load ground truth imgs: N x H x W x 3

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        x = self.image_x[idx]
        y = self.image_y[idx]
        return [x, y]


def load_one_image(path):
    img = io.imread(path)
    return torch.FloatTensor(img)


def data_walk(walk_dir):
    img_names = []
    for file in os.listdir(walk_dir):
        if file.endswith(".jpg"):
            img_names.append(file)
    return img_names


def load_all_images(image_dir):
    img_names = data_walk(image_dir)
    all_imgs = []
    for name in img_names:
        path = image_dir + name
        img = load_one_image(path)
        all_imgs.append(img)
    return torch.FloatTensor(all_imgs)


def get_patches(images):
    N, height, width, channels = images.shape
    n_h = height // patch_m
    n_w = width // patch_m
    patches = np.zeros((N, n_h, n_w, patch_m, patch_m, channels))
    for i in range(N):
        for h in range(n_h):
            for w in range(n_w):
                patches[i, h, w] = images[i, n_h*patch_m:(n_h+1)*patch_m, n_w*patch_m:(n_w+1)*patch_m, :]
    patches = patches.reshape((N*n_h*n_w, patch_m, patch_m, channels))
    return torch.FloatTensor(patches)

