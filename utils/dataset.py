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
        original_path = "../data/train_original/"  # where to load original images
        beautify_path = "../data/train_beautified"  # where to load beautified imgs
        self.image_x = load_all_images(beautify_path)  # load beautified images: N x H x W x 3
        image_y = load_all_images(original_path)      # load ground truth imgs: N x H x W x 3
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
        original_path = "../data/test_original/"  # where to load original images
        beautify_path = "../data/test_beautified" # where to load beautified imgs
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
    return np.array(img)


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
	return np.array(all_imgs)
