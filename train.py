import argparse
import os
import logging

import yaml
import torch
from torch.utils import data

from module import CRN, _pca_
# import evaluator as _evaluator
from utils import registry
from utils import dataset as _dataset
# from utils import optimizer as _optimizer
# from utils import lr_scheduler as _lr_scheduler
# from utils import loss as _loss
from utils import arg



def train(param):
    # TODO
    # Use PCA part to get a test dataset:
    # DATASET:                  1. train data: the beautified images
    # (Based on PCA Results ->) 2. The components of the difference between the beautified images and the ground truth
    # Check the size of dataset images, decide whether to use batch operation
    # return a tensor of the eigenvalues
    # DRAFT:

    train_dataset = registry.create('Dataset', param["train_dataset"]["name"])(**param["train_dataset"]["kwargs"])
    valid_dataset = registry.create('Dataset', param["valid_dataset"]["name"])(**param["valid_dataset"]["kwargs"])
    train_data_loader = data.DataLoader(train_dataset, **param["loader"])
    valid_data_loader = data.DataLoader(valid_dataset, **param["loader"])
    # pin_memory=param.use_gpu,

    # eigenvalues = train_dataset.get_()
    # model = torch.nn.DataParallel(registry.create('Network', param.network.name)(**param.network.kwargs))
    model = CRN.CRN(**param["network"]["kwargs"])

    criterion = registry.create('Loss', param["loss"]["name"])(**param["loss"]["kwargs"])
    optimizer = registry.create('Optimizer', param["optimizer"]["name"])(model.parameters(), **param["optimizer"]["kwargs"])
    lr_scheduler = registry.create('LRScheduler', param["lr_scheduler"]["name"])(optimizer, **param["lr_scheduler"]["kwargs"])

    pca = _pca_.PCA(param["PCA"]["kwargs"])

    # TODO: checkpoint

    # if param.use_gpu:
    #     model = model.cuda()

    for epoch in range(int(param["epoch"])):
        for inputs, target in train_data_loader:
            optimizer.zero_grad()

            ground_truth = pca.get_components(inputs, targets, True)

            output = model(inputs)

            # how to get the truth
            # ground_truth = pca.get_components(inputs, targets, False)
            # how to get an img:
            # output = model(inputs)
            # img = pca.generate_an_img(output, inputs)

            loss = criterion(output, ground_truth)

            loss.backward()
            optimizer.step()

        # model.eval()
        # @TODO: EVALUATE!!!
        # valid_data_loader...

        # model.train()

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        },
            os.path.join("TODO!!!")
        )
        logging.debug('saving model done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_params', '-lp', default='default', help='The global parameters for the program to load, from param/ folder')
    parser.add_argument('--overwrite_param', '-op', action=arg.StoreDictKeyPair, help='The parameters to override')

    args = parser.parse_args()
    # print(args.overwrite_param)
    param = yaml.load(open(os.path.join('param', '{}.yml'.format(args.load_params))))
    # print(param)
    arg.update(param, args.overwrite_param)
    print(param)
    train(param)
