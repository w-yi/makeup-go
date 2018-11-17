import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class pca(PCA):
	def __init__(self, vectorized_data=None):
		print("Input vectorized data matrix :{}".format(vectorized_data.shape))
		self.data = vectorized_data
		self._n, self._n_components, self._channels = vectorized_data.shape
		super(pca, self).__init__(n_components=self._n_components)

	def get_principle_components(self):
		# get principle vectors from all channels
		all_principle_components = []
		for i in range(self._channels):
			self.fit_transform(self.data[:,:,i])
			all_principle_components.append(self.components_)
		return np.array(all_principle_components)


def get_errors(ground_image, predict_image, ground_principle_components):
	for i in range(len(ground_principle_components)):
		n = len(ground_principle_components[i])
		ground_errors = np.zeros(n)
		predict_errors = np.zeros(n)
		# for j in range(n):
		# 	ground_errors[j] = np.dot(ground_principle_components[i, j], 


