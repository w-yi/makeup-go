import argparse
import os
import logging

import yaml
import torch
from torch.utils import data

from utils import registry
from utils import dataset as _dataset

from utils import arg


def main(param):

    _ = registry.create('Dataset', param["dataset"]["name"])(**param["dataset"]["kwargs"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_params', '-lp', default='default', help='The global parameters for the program to load, from param/ folder')

    args = parser.parse_args()
    # print(args.overwrite_param)
    param = yaml.load(open(os.path.join('param', '{}.yml'.format(args.load_params))))
    # print(param)
    print(param)

    main(param)
