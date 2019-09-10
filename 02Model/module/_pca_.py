import torch


patch_m = 11


class PCA():
    def __init__(self, momentum=0.1, n_top=7, kernel=patch_m):
    # def __init__(self):
        with torch.no_grad():
            # dim = kernel*kernel*3
            # self.kernel = kernel
            # self.n_top = n_top
            # self.mean = torch.zeros(1, dim)
            # self.proj = torch.eye(dim)
            self.eigenvalues = (torch.load("R.t")).float().cuda()
            self.eigenvectors = torch.load("Q.t")[:, :7].float().cuda()
            # print("eigenvectors ")
            self.mean = (torch.load("mean.t")).float().cuda()
            # .view(3,11,11)
            # self.momentum = momentum

    # def add_truth(self, X):
    #     with torch.no_grad():
    #         n, _ = X.size()
    #         self.mean = self.momentum * self.mean + (1-self.momentum) * torch.mean(X, 0, keepdim=True)  # 1 x d
    #         X = X - self.mean
    #         self.proj = self.momentum * self.proj + (1-self.momentum) * X.t() @ X / n  # d x d
    #         val, vec = torch.symeig(self.proj, True)
    #         self.eigenvalues = val[-self.n_top-1:].flip(0)
    #         self.eigenvectors = torch.stack(
    #             [vec[-1-i].flip(0).view(3, self.kernel, self.kernel) for i in range(self.n_top)]
    #         )


    def get_components(self, x, y, train):
    # def get_components(self, x, y, train):
        with torch.no_grad():
            # print('mean shape:', self.mean.shape)
            # print('y shape', y.shape)
            # print('x shape', x.shape)
            e = y - x 
            # torch.save(e, "e.t")
            patches, sz = get_patches(e)
            centered_patches = patches - self.mean
            
            components = []
            for i in range(7):
                # print('eigenvectors shape', self.eigenvectors.shape)
                # print('patches shape', patches.shape)
                # print('evectors shape', self.eigenvectors[:, i].shape)
                # temp = patches @ self.eigenvectors[:, i]
                # print('lalala')

                # print('temp shape', temp.shape)
                # print('mean shape', self.mean.shape)
                b = self.eigenvectors[:, i].view(-1,1)
                # print("b shape", b.shape)
                # print("patches - mean", (patches - self.mean).shape)
                temp = (centered_patches) @ b @ b.t()
                # components += [combine_patches(temp, sz)]
                components += [temp]
            
            components += [centered_patches - torch.sum(torch.stack(components), 0)]
            return torch.stack(
                [val ** (-1) * combine_patches(component, sz) for val, component in zip(self.eigenvalues, components)], 1
            )

    def generate_img(self, components, x):
        with torch.no_grad():
            results = []
            sz = None
            for val, component in zip(self.eigenvalues, components):
                patches, sz = get_patches(val * component.unsqueeze(0))
                # temp = combine_patches(patches, sz)
                results += [patches]
            # all_mean = torch.zeros(e.shape)
            
            e = torch.sum(torch.stack(results), 0) + self.mean
            e = combine_patches(e, sz)
            # torch.save(e, "e_re.t")


            ####

            # print('########################')
            # print(e)
            #####


            return e + x


def get_patches(images):
    with torch.no_grad():
        N, channels, height, width = images.shape
        assert channels is 1 or channels is 3, "The image has {} channels (expect 1 or 3)".format(channels)
        if channels == 1:
            images = torch.stack([images, images, images], 0)
            channels = 3
        n_h = height // patch_m
        n_w = width // patch_m

        patches = torch.zeros((N, n_h, n_w, channels, patch_m, patch_m))

        # original
        # patches = torch.zeros((N, channels, n_h, n_w, patch_m, patch_m))
        for i in range(N):
            for h in range(n_h):
                for w in range(n_w):
                    # print(patches[i, :, h, w].shape, images[i, :, h*patch_m:(h+1)*patch_m, w*patch_m:(w+1)*patch_m].shape)
                    patches[i, h, w, :] = images[i, :, h*patch_m:(h+1)*patch_m, w*patch_m:(w+1)*patch_m]
        # patches = np.transpose(patches, (0, 2, 3, 1, 4, 5))
        sz = patches.shape
        patches = patches.view(N, n_h, n_w, -1)
        return patches, sz


def combine_patches(patches, sz):
    patches = patches.view(sz)
    N, n_h, n_w, channels, patch_m, patch_m = patches.shape
    # print("size:", sz)
    height = n_h * patch_m
    width = n_w * patch_m
    images = torch.zeros(N, channels, height, width)
    for i in range(N):
        for h in range(n_h):
            for w in range(n_w):
                # print(patches[i, :, h, w].shape, images[i, :, h*patch_m:(h+1)*patch_m, w*patch_m:(w+1)*patch_m].shape)
                # images[i, :, h * patch_m:(h + 1) * patch_m, w * patch_m:(w + 1) * patch_m] = patches[i, :, h, w]
                images[i, :, h*patch_m:(h+1)*patch_m, w*patch_m:(w+1)*patch_m] = patches[i, h, w, :]
    return images
