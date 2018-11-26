import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import convolve


class pca(PCA):
    def __init__(self, data=None):
        print("Input data matrix :{}".format(data.shape))
        self.shape = data.shape
        N, H, W, C = self.shape
        vectorized_data = data.reshape((N, -1))
        print("Vectorized data :{}".format(vectorized_data.shape))
        self.data = vectorized_data
        self._n, self._n_components = vectorized_data.shape
        # m = int(np.sqrt(self._n_components))
        # if m**2 != self._n_components:
        #     raise RuntimeError("Non-square patches found! Use square patch instead.")
        super(pca, self).__init__(n_components=self._n_components) 

    def get_evec_eval(self):
        # get principle vectors from all channels
        # all_principle_components = []
        # for i in range(self._channels):
        self.fit_transform(self.data)
        all_principle_components = self.components_
        e_variances = self.explained_variance_ 
        return np.array(all_principle_components), np.sqrt(e_variances)


    def get_pca_convolution_kernel(self):
        all_principle_components, _ = self.get_evec_eval()
        # m = int(np.sqrt(self._n_components))
        N, H, W, C = self.shape
        all_principle_components = all_principle_components.reshape((H*W*C,H,W,C))
        all_principle_components_rotated = np.rot90(all_principle_components, 2, axes=(2,3))
        return all_principle_components_rotated


def get_errors(ground_images, predict_images, ground_rotated_kernels):
    for i in range(len(ground_rotated_kernels)):
        m = len(ground_rotated_kernels[i,0])
        n = len(ground_images)
        ground_errors = np.zeros(n)
        predict_errors = np.zeros(n)
        # img_convolve = convolve(ground_images, kernel_rotate, mode="constant", cval=0.0)
        # img_convolve = convolve(img_convolve, kernel_rotate,  mode="constant", cval=0.0)


# if __name__ == "__main__":
#     data = np.random.rand(10000,11,11,3)
#     my_pca = pca(data)
#     evecs, evals = my_pca.get_evec_eval()
#     print(evecs.shape)
#     print(evals[0:10])

