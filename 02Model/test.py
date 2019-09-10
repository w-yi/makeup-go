import argparse
import os
import logging
import pickle
import numpy as np

import yaml
import torch
from torch.utils import data
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from module import CRN, _pca_
# import evaluator as _evaluator
from utils import registry
from utils import dataset as _dataset
from utils import optimizer as _optimizer
from utils import lr_scheduler as _lr_scheduler
from utils import loss as _loss
from utils import arg


from PIL import Image
from scipy.misc import imread
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt



def train(param):
    # train_dataset = registry.create('Dataset', param["train_dataset"]["name"])(**param["train_dataset"]["kwargs"])
    # valid_dataset = registry.create('Dataset', param["valid_dataset"]["name"])(**param["valid_dataset"]["kwargs"])
    # train_data_loader = data.DataLoader(train_dataset, **param["loader"])
    pca = _pca_.PCA(**param["PCA"]["kwargs"])

    # model_names = ["model_CRN_6_110.pkl",
    # "model_CRN_6_120.pkl",
    # "model_CRN_6_130.pkl",
    # "model_CRN_6_140.pkl",
    # "model_CRN_6_150.pkl"]

    model_names = ["final_model.pkl"]

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    selected_ids = [3, 903, 1015, 1012, 1008]

    for model_id, model_name in enumerate(model_names):

        model = pickle.load(open("./models/" + model_name, "rb"))
        model = model.cuda()

        print("starting validating...")

        parts = model_name.split(".")
        # model_path = "./output_beautified/" + parts[0]
        model_path = "./output"

        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        if param["valid"]:
            with torch.no_grad():
                print(param["valid"])
                for i in selected_ids:
                    # path = param["valid"] + "/beautified ({}).jpg".format(i)
                    # inputs = img_to_tensor(path)
                    # path = "./data/EECS442_Makeup_Go/result_original/original ({}).jpg".format(i)
                    # targets = img_to_tensor(path)
                    # inputs.unsqueeze_(0)
                    # targets.unsqueeze_(0)

                    # ground_truth = pca.get_components(inputs, targets, True)
                    # print("generating one image...")
                    # ground_truth = ground_truth[0]
                    
                    # print("gound truth size ", ground_truth.shape)

                    # for j in range(8):
                    #     img = np.array(ground_truth[j]).transpose((1,2,0))
                    #     img[img>255] = 255
                    #     img[img<0] = 0
                    #     cm = None
                    #     fn = model_path + "/result_{}_{}.jpg".format(i, j)
                    #     save_image(img, cm, fn)

                    # path = param["valid"] + "/beautified({}).jpg".format(i)
                    # x = img_to_tensor(path).cuda()
                    # x.unsqueeze_(0)
                    # output = model(x)
                    # img = pca.generate_img(output, x)
                    # np_img = np.array(img[0])
                    # np_img[np_img>255] = 255
                    # np_img[np_img<0] = 0
                    # np_img = np_img.transpose((1, 2, 0)).astype(np.uint8)
                    # plt.close()
                    # plt.imshow(np_img)
                    # plt.savefig(model_path + "/result_{}.jpg".format(i))

                    path = param["valid"] + "/beautified ({}).jpg".format(i)
                    x = img_to_tensor(path).cuda()
                    sz = x.shape
                    x.unsqueeze_(0)
                    output = torch.squeeze(model(x), 0)
                    print("generating one image...")
                    img = pca.generate_img(output, x)
                    np_img = np.array(img[0])
                    np_img[np_img>255] = 255
                    np_img[np_img<0] = 0
                    np_img = np_img.transpose((1, 2, 0)).astype(np.uint8)
                    save_image(np_img, None, model_path + "/result_{}.jpg".format(i))
                    # plt.close()
                    # plt.imshow(np_img)
                    # plt.savefig(model_path + "/result_{}.jpg".format(i))


def img_to_tensor(path):
    image = imread(path)
    image = image[:46 * 11, :46 * 11, :]
    image = np.transpose(image, (2,0,1))
    x = torch.tensor(image).cuda()
    # x = TF.to_tensor(image).cuda()
    return x.float()


def save_image(data, cm, fn):
    cmap = plt.cm.jet 
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
#     data = data.astype(np.uint8)
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(fn, dpi = height, cmap=cmap) 
    plt.close()


if __name__ == '__main__':

    

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_params', '-lp', default='default', help='The global parameters for the program to load, from param/ folder')
    parser.add_argument('--overwrite_param', '-op', action=arg.StoreDictKeyPair, help='The parameters to override')

    args = parser.parse_args()
    # print(args.overwrite_param)
    param = yaml.load(open(os.path.join('param', '{}.yml'.format(args.load_params))))
    print(param["valid"])

    # print(param)
    arg.update(param, args.overwrite_param)
    train(param)
