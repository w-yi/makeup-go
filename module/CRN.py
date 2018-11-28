import torch
from torch import nn


IN_CHANNEL = 3


class _debug(nn.Module):
    def __init__(self):
        super(_debug, self).__init__()

    @staticmethod
    def forward(x):
        print(x.shape)
        return x


class CRN(nn.Module):
    def __init__(self, n_top=7, kernel1=3, kernel2=3, channel1=56, channel2=12, channel3=64, nonlinear="PReLU"):
        # channel3???????????64, eigenvalues
        super(CRN, self).__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.nonlinear = nonlinear

        self.common_network = self.generate_commonnetwork()
        self.subnetwork_list = [self.generate_subnetwork() for _ in range(n_top+1)]

        # self._initialize_weights()

    def forward(self, x):
        features = self.common_network(x)
        components = [sub_network(features) for sub_network in self.subnetwork_list]
        return torch.stack(components, dim=1)

    def get_nonlinear(self):
        return{
            "PReLU": nn.PReLU(),  # num_parameters=1, init=0.25
            "ReLU":  nn.ReLU()
        }[self.nonlinear]

    def generate_commonnetwork(self):
        return nn.Sequential(
            nn.Conv2d(IN_CHANNEL, self.channel1, self.kernel1, padding=1),
            self.get_nonlinear(),
            nn.Conv2d(self.channel1, self.channel1, self.kernel1, padding=1),
            self.get_nonlinear(),
            nn.Conv2d(self.channel1, self.channel1, self.kernel1, padding=1),
            self.get_nonlinear(),
        )

    def generate_subnetwork(self):
        return nn.Sequential(
            # shrinking
            nn.Conv2d(self.channel1, self.channel2, self.kernel2, padding=1),
            # stack conv with nonlinear
            self.get_nonlinear(),
            nn.Conv2d(self.channel2, self.channel2, self.kernel2, padding=1),
            self.get_nonlinear(),
            nn.Conv2d(self.channel2, self.channel2, self.kernel2, padding=1),
            self.get_nonlinear(),
            nn.Conv2d(self.channel2, self.channel2, self.kernel2, padding=1),
            self.get_nonlinear(),
            # expending
            nn.Conv2d(self.channel2, self.channel3, self.kernel2, padding=1),
            self.get_nonlinear(),
            nn.Conv2d(self.channel3, IN_CHANNEL, self.kernel2, padding=1),
        )

