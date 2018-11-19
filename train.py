import argparse
import os
import logging

import yaml
import torch
from torch.utils import data

from module import CRN
# import evaluator as _evaluator
from utils import registry
# from utils import dataset as _dataset
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

    train_dataset = registry.create('Dataset', param["train_dataset"]["name"])()
    valid_dataset = registry.create('Dataset', param["valid_dataset"]["name"])()
    train_data_loader = data.DataLoader(train_dataset, param.loader.batch_size, shuffle=param.loader.shuffle,
                                        num_workers=param.loader.num_workers, pin_memory=param.use_gpu,
                                        drop_last=param.loader.drop_last)
    valid_data_loader = data.DataLoader(valid_dataset, param.loader.batch_size, shuffle=param.loader.shuffle,
                                        num_workers=param.loader.num_workers, pin_memory=param.use_gpu, drop_last=False)

    eigenvalues = train_dataset.get_()
    # model = torch.nn.DataParallel(registry.create('Network', param.network.name)(**param.network.kwargs))
    model = CRN.CRN(eigenvalues=eigenvalues, **param["network"]["kwargs"])

    criterion = registry.create('Loss', param["loss"]["name"])(**param["loss"]["kwargs"])
    optimizer = registry.create('Optimizer', param["optimizer"]["name"])(model.parameters(), **param["optimizer"]["kwargs"])
    lr_scheduler = registry.create('LRScheduler', param["lr_scheduler"]["name"])(optimizer, **param["lr_scheduler"]["kwargs"])

    # TODO: checkpoint

    # if param.use_gpu:
    #     model = model.cuda()

    for epoch in range(int(param["epoch"])):
        for inputs, target in train_data_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

        model.eval()
        # @TODO: EVALUSTE!!!
        # valid_data_loader...
        model.train()

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
