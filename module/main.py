import numpy as np
from pca import pca


def main():
	data = np.random.rand(100,9,3)
	my_pca = pca(data)
	principle_components = my_pca.get_principle_components()
	print(principle_components.shape)


if __name__ == '__main__':
	main()
