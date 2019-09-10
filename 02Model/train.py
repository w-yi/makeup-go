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
    # TODO
    # Use PCA part to get a test dataset:
    # DATASET:                  1. train data: the beautified images
    # (Based on PCA Results ->) 2. The components of the difference between the beautified images and the ground truth
    # Check the size of dataset images, decide whether to use batch operation
    # return a tensor of the eigenvalues
    # DRAFT:

    train_dataset = registry.create('Dataset', param["train_dataset"]["name"])(**param["train_dataset"]["kwargs"])
    # valid_dataset = registry.create('Dataset', param["valid_dataset"]["name"])(**param["valid_dataset"]["kwargs"])
    train_data_loader = data.DataLoader(train_dataset, **param["loader"])
    # valid_data_loader = data.DataLoader(valid_dataset, **param["loader"])
    # pin_memory=param.use_gpu,

    # eigenvalues = train_dataset.get_()
    # model = torch.nn.DataParallel(registry.create('Network', param.network.name)(**param.network.kwargs))
    model = CRN.CRN(**param["network"]["kwargs"])


    criterion = registry.create('Loss', param["loss"]["name"])(**param["loss"]["kwargs"])
    optimizer = registry.create('Optimizer', param["optimizer"]["name"])(model.parameters(), **param["optimizer"]["kwargs"])
    lr_scheduler = registry.create('LRScheduler', param["lr_scheduler"]["name"])(optimizer, **param["lr_scheduler"]["kwargs"])

    pca = _pca_.PCA(**param["PCA"]["kwargs"])

    # TODO: checkpoint

    # if param.use_gpu:
    #     model = model.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = model.cuda()

    load_previous = False
    if load_previous:
        pca = pickle.load(open("trained_models/model_PCA_9_0.pkl", "rb"))
        model = pickle.load(open("trained_models/model_CRN_9_0.pkl", "rb"))

    is_train = True

    g_counter = 0

    if is_train:
        save_dir = "./models"
        # idx = int(os.listdir(save_dir)[-1][5:]) + 1
        save_dir = "./models/model_PCA_modified"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir += "/"

        if not os.path.exists("./output_PCA_modified"):
            os.mkdir("./output_PCA_modified")

        for epoch in range(int(param["epoch"])):
            print("epoch:", epoch)
            counter = 0
            for inputs, targets in train_data_loader:
                lr_scheduler.step()
                optimizer.zero_grad()

                ground_truth = pca.get_components(inputs, targets, True)
                # print('########')
                # print(ground_truth.shape)
                # for current_data in ground_truth:
                #     for img in current_data:
                #         save_image(img/torch.max(img)*255, 'decomposed_ground_truth/' + str(g_counter) + '.jpg')
                #         g_counter += 1
                # print('########')
                # print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
                # print(ground_truth)
                # print('-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-')

                # print("get")

                output = model(inputs)
                # print(output)
                # print("model")
                # with torch.no_grad():
                #     img = pca.generate_img(output[0], inputs[0])
                #     save_image(img, "output_PCA_modified/train/result_{}_{}_before.jpg".format(epoch, counter))

                loss = criterion(output, ground_truth)

                if counter % 10 == 0:
                    # if not torch.isnan(loss):
                    #     # Fixme change the epoch +10 back
                    # with open(save_dir + 'model_PCA_{}_{}.pkl'.format(epoch+10, counter), 'wb') as out:
                    #     pickle.dump(pca, out, pickle.HIGHEST_PROTOCOL)

                    with open(save_dir + 'model_CRN_{}_{}.pkl'.format(epoch, counter), 'wb') as out:
                        pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)

                    print("loss {}:".format(counter), loss.item(), "\tlr:",lr_scheduler.get_lr()[0])

                    print("starting validating...")
                    if param["valid"]:
                        with torch.no_grad():
                            # image = Image.open(param["valid"])
                            # x = TF.to_tensor(image).cuda()
                            x = img_to_tensor(param["valid"])
                            sz = x.shape
                            x.unsqueeze_(0)
                            output = torch.squeeze(model(x), 0)
                            print("generating one image...")
                            img = pca.generate_img(output, x)
                            np_img = np.array(img[0])
                            # print('before conversion')
                            # print('@@@@@@@@')
                            # print(np_img)
                            np_img[np_img>255] = 255
                            np_img[np_img<0] = 0
                            np_img = np_img.transpose((1, 2, 0)).astype(np.uint8)
                            plt.close()
                            plt.imshow(np_img)
                            plt.savefig("output_PCA_modified/result_{}_{}.jpg".format(epoch, counter))
                            # save_image(img, "output_PCA_modified/result_{}_{}.jpg".format(epoch, counter))

                loss.backward()

                # perform gradient clipping
                clip_grad_norm_(model.parameters(), 2)

                optimizer.step()
                # with torch.no_grad():
                #     output = torch.squeeze(model(inputs[0].unsqueeze_(0)), 0)
                #     img = pca.generate_img(output, inputs[0]).view(sz)
                #     save_image(img, "output_PCA_modified/train/result_{}_{}_after.jpg".format(epoch, counter))
                counter += 1

            # model.eval()
            # @TODO: EVALUATE!!!
            # valid_data_loader...
            # how to get the truth
            # ground_truth = pca.get_components(inputs, targets, False)
            # how to get img:
            # output = model(inputs)
            # img = pca.generate_img(output, inputs)

            # model.train()

            # torch.save({
            #     'model': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'lr_scheduler': lr_scheduler.state_dict()
            # },
            #     os.path.join("TODO!!!")
            # )
            # logging.debug('saving model done')

            # print("starting validating...")
            # if param["valid"]:
            #     with torch.no_grad():
            #         image = Image.open(param["valid"])
            #         x = TF.to_tensor(image).cuda()
            #         sz = x.shape
            #         x.unsqueeze_(0)
            #         output = torch.squeeze(model(x), 0)
            #         print("generating one image...")
            #         img = pca.generate_img(output, x).view(sz)
            #         save_image(img, "output_PCA_modified/result_{}.jpg".format(epoch))
        
    else:
        print("starting validating...")
        
        if param["valid"]:
            with torch.no_grad():
                image = Image.open(param["valid"])
                parts = param["valid"].split("/")
                parts = parts[-1].split(".")
                parts = parts[0].split("_")
                name = parts[0]
                x = TF.to_tensor(image).cuda()
                sz = x.shape
                x.unsqueeze_(0)
                output = model(x)
                print(output.shape)
                output = torch.squeeze(output,0)
                print(output.shape)
                print("generating one image...")
                img = pca.generate_img(output, x).view(sz)
                print(img)
                save_image(img, "output/{}_result.jpg".format(name))


def img_to_tensor(path):
    image = imread(path)
    image = image[:46 * 11, :46 * 11, :]
    image = np.transpose(image, (2,0,1))
    x = torch.tensor(image).cuda()
    # x = TF.to_tensor(image).cuda()
    return x.float()


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
    # print(param)
    # if param["valid"]:
    #     image = Image.open(param["valid"])
    #     x = TF.to_tensor(image).cuda()
    #     sz = x.shape
    #     x.unsqueeze_(0)
    #     y = x.view(sz)
    #     save_image(y, "final_result.jpg")
    train(param)
