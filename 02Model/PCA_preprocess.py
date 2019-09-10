
import argparse
import os
import cv2
import numpy as np


def main(args):
    beautify = [f for f in os.listdir(args.beautify_path)]
    original = [f for f in os.listdir(args.original_path)]
    beautify.sort()
    original.sort()

    len = args.len
    e = np.concatenate([get_patches(
        np.transpose((one_img(os.path.join(args.original_path,original[i])) - one_img(os.path.join(args.beautify_path, beautify[i]))), (2, 0, 1))
    ) for i in range(len)])
    mean = np.mean(e, 0, keepdims=True)
    np.save("mean", mean)
    e = mean - e
    cov = e.T @ e
    Q, R = simultaneous_power_iteration(cov)
    np.save("Q", Q)
    np.save("R", R)




def one_img(filename):
    im = cv2.imread(filename, 1)
    return im


def get_patches(images, patch_m=11):
    images = np.expand_dims(images, axis=0)
    N, channels, height, width = images.shape
    n_h = height // patch_m
    n_w = width // patch_m
    patches = np.zeros((N, channels, n_h, n_w, patch_m, patch_m))
    for i in range(N):
        for h in range(n_h):
            for w in range(n_w):
                # print(patches[i, :, h, w].shape, images[i, :, h*patch_m:(h+1)*patch_m, w*patch_m:(w+1)*patch_m].shape)
                patches[i, :, h, w] = images[i, :, h*patch_m:(h+1)*patch_m, w*patch_m:(w+1)*patch_m]
    patches = np.reshape(patches, (N*n_h*n_w, -1))
    return patches


def simultaneous_power_iteration(X, r=8):
    n, _ = X.shape
    Q = np.zeros((n, r))
    for i in range(r):
        Q[i][i] = 1
    R = np.zeros((r, r))
    for i in range(500):
        e = Q
        Q, R = np.linalg.qr(X @ Q)
        e = e - Q
        e = e*e
        e = np.sum(e)
        print(e)
    return Q, R




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', '-o', help='directory to original images')
    parser.add_argument('--beautify_path', '-b', help='directory to beautify images')
    parser.add_argument('--len', '-l', default=10, type=int, help='data size')

    args = parser.parse_args()
    main(args)