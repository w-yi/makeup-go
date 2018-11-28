import torch


patch_m = 11


class PCA():
    def __init__(self, momentum=0.1, n_top=7, kernel=patch_m):
        dim = kernel*kernel*3
        self.kernel = kernel
        self.n_top = n_top
        self.mean = torch.zeros(1, dim)
        self.proj = torch.eye(dim)
        self.eigenvalues = torch.zeros(n_top+1)
        self.eigenvectors = torch.zeros(n_top, 3, kernel, kernel)
        self.momentum = momentum

    def add_truth(self, X):
        n, _ = X.size()
        self.mean = self.momentum * self.mean + (1-self.momentum) * torch.mean(X, 0, keepdim=True)  # 1 x d
        X = X - self.mean
        self.proj = self.momentum * self.proj + (1-self.momentum) * X.t() @ X / n  # d x d
        val, vec = torch.symeig(self.proj, True)
        self.eigenvalues = val[-self.n_top-1:].flip(0)
        self.eigenvectors = torch.stack(
            [vec[-1-i].flip(0).view(3, self.kernel, self.kernel) for i in range(self.n_top)]
        )

    def get_components(self, x, y, train):
        e = y - x
        if train:
            self.add_truth(get_patches(e).view(-1, self.kernel * self.kernel * 3))  # n x d
        components = [torch.nn.functional.conv2d(
            e, weight.view(3, 1, self.kernel, self.kernel), padding=(self.kernel-1)//2, groups=3
        ) for weight in self.eigenvectors]
        components = [torch.nn.functional.conv2d(
            comp, weight.view(3, 1, self.kernel, self.kernel), padding=(self.kernel - 1) // 2, groups=3
        ) for weight, comp in zip(self.eigenvectors, components)]
        components += [e - torch.sum(torch.stack(components), 0)]
        return torch.stack(
            [val ** (-0.5) * component for val, component in zip(self.eigenvalues, components)], 1
        )

    def generate_img(self, components, x):
        components = [val**(0.5) * component for val, component in zip(self.eigenvalues, components)]
        e = torch.sum(torch.stack(components), 0) + self.mean
        return e + x


def get_patches(images):
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
    patches = patches.view(N*n_h*n_w, channels, patch_m, patch_m)
    return patches
