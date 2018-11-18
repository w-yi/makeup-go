import sys
sys.path.append('../module')

import numpy as np
from pca import pca
from scipy.ndimage import convolve


def test():
    img = np.array(np.random.rand(10,10)*256, dtype=int)
    print(img)
    img_pad = np.pad(img, ((1,1),(1,1)), "constant")
    kernel = np.array(np.random.rand(4,4)*4, dtype=int)
    print(img_pad)
    print("==========Two convolutions===========")
    kernel_rotate = np.rot90(kernel, 2)
    img_convolve = convolve(img, kernel_rotate, mode="constant", cval=0.0)
    # print(img_convolve)
    # kernel_convolve = np.matmul(kernel, kernel)

    img_convolve1 = convolve(img_convolve, kernel_rotate,  mode="constant", cval=0.0)
    # convolve(img_convolve, kernel, mode="valid")
    print(img_convolve1)
    # print("======================================")

    # img_convolve1 = np.dot(kernel.reshape((1,16)), img.reshape((16,1)))
    # img_convolve1 = np.matmul(kernel.reshape((16,1)), img_convolve1)
    # print(img_convolve1.reshape(4,4))

    # print("==========One convolution============")
    # img_convolve2 = np.dot(kernel.reshape((16,1)), kernel.reshape((1,16)))
    # img_convolve2 = np.matmul(img_convolve2, img.reshape((16,1)))

    # kernel_convolve = np.matmul(kernel, kernel_rotate)
    # #convolve(kernel, kernel_rotate, mode="constant", cval=0.0)
    # img_convolve2 = convolve(img, kernel_convolve, mode="constant", cval=0.0)
    # print(img_convolve2.reshape((4,4)))

    # norm = np.max(img_convolve1 - img_convolve2)
    # print(img_convolve1 - img_convolve2)
    # print(kernel.reshape(4))
    # print(kernel_rotate.reshape(4))


def main():
    data = np.random.rand(10000,121,3)
    my_pca = pca(data)
    kernels = my_pca.get_pca_convolution_kernel()
    print(kernels.shape)


if __name__ == '__main__':
    main()
    # test()
