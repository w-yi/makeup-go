import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import convolve


class pca(PCA):
	def __init__(self, vectorized_data=None):
		print("Input vectorized data matrix :{}".format(vectorized_data.shape))
		self.data = vectorized_data
		self._n, self._n_components, self._channels = vectorized_data.shape
		m = int(np.sqrt(self._n_components))
		if m**2 != self._n_components:
			raise RuntimeError("Non-square patches found! Use square patch instead.")
		super(pca, self).__init__(n_components=self._n_components)

	def get_principle_components(self):
		# get principle vectors from all channels
		all_principle_components = []
		for i in range(self._channels):
			self.fit_transform(self.data[:,:,i])
			all_principle_components.append(self.components_)
		return np.array(all_principle_components)

	def get_pca_convolution_kernel(self):
		all_principle_components = self.get_principle_components()
		m = int(np.sqrt(self._n_components))
		all_principle_components = all_principle_components.reshape((self._channels,self._n_components,m,m))
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
