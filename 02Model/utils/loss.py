from utils import registry
import torch.nn
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


registry.register('Loss', 'CrossEntropy')(torch.nn.CrossEntropyLoss)
registry.register('Loss', 'MSE')(torch.nn.MSELoss)

@registry.register('Loss', 'PatchWeightSquare')
class PatchWeightSquare(_Loss):

    def __init__(self, basis):
        super(PatchWeightSquare, self).__init__()
        self.basis = (torch.load(basis)).float().cuda()

    def forward(self, input, target):
        loss = torch.tensor(0).float()
        for i in range(7):
            x = get_patches(input[:, i, :, :, :])
            y = get_patches(target[:, i, :, :, :])

            w_x = x @ self.basis[:,i]
            w_y = y @ self.basis[:,i]
            w_diff = w_x - w_y

            loss += torch.sum(w_diff*w_diff)/torch.sum(w_x*w_x)
        sz = input[:, 7, :, :, :].shape
        mse = F.mse_loss(input[:, 7, :, :, :], target[:, 7, :, :, :])
        # print("loss:!!!!", loss, ",",mse)
        loss += mse/(F.mse_loss(input[:, 7, :, :, :], torch.zeros(sz)))

        return loss


def get_patches(images, patch_m=11):
    with torch.no_grad():
        N, channels, height, width = images.shape
        assert channels is 1 or channels is 3, "The image has {} channels (expect 1 or 3)".format(channels)
        if channels == 1:
            images = torch.stack([images, images, images], 0)
            channels = 3
        n_h = height // patch_m
        n_w = width // patch_m
        patches = torch.zeros((N, channels, n_h, n_w, patch_m, patch_m))
        for i in range(N):
            for h in range(n_h):
                for w in range(n_w):
                    # print(patches[i, :, h, w].shape, images[i, :, h*patch_m:(h+1)*patch_m, w*patch_m:(w+1)*patch_m].shape)
                    patches[i, :, h, w] = images[i, :, h*patch_m:(h+1)*patch_m, w*patch_m:(w+1)*patch_m]
        patches = patches.view(N*n_h*n_w, -1)
        return patches


#n,8,3,512,512