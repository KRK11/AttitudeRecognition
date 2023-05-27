'''
**************************************************
@File   ：AttitudeRecognition -> image_utils
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/27 11:49
**************************************************
'''
import random

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def load_image(path=None):
    return cv2.imread(path, cv2.IMREAD_COLOR) # BGR


def to_torch(ndarray):
    return torch.from_numpy(ndarray)


def to_numpy(tensor):
    return tensor.detach().clone().cpu().numpy()


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()

    if img.max() > 1:
        img /= 255
    return img


def generate_heatmap(heatmap, pt, sigma):
    x, y = int(pt[0]), int(pt[1])
    x = min(x, 47)
    y = min(y, 63)
    heatmap[y][x] = 1
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    heatmap /= am / 255
    return heatmap


def generate_label(keypoints, num_class, gauss_kernel, out_shape=(64, 48)):
    heatmap = np.zeros((num_class, out_shape[0], out_shape[1]))
    for i in range(num_class):
        if keypoints[i, 2] > 0:
            heatmap[i] = generate_heatmap(heatmap[i], keypoints[i, :2], gauss_kernel)
    return torch.Tensor(heatmap)


def show_heatmap(heatmap, image, keypoints):
    fig = plt.figure(figsize=(50, 50))
    for i in range(heatmap.shape[0]):
        plt.subplot(9, 4, i * 2 + 1)
        plt.imshow(heatmap[i], cmap='hot', interpolation='nearest')
        plt.colorbar()

        x, y, _ = keypoints[i]
        plt.subplot(9, 4, i * 2 + 2)
        plt.imshow(image)
        plt.plot(x, y, 'yo')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    score = np.zeros((20, 20))
    score = generate_heatmap(score, (10, 10), (15, 15))
    print(score[10, 10])
    print(score)
    # score = np.zeros((17, 64, 48))
    # for i in range(17):
        # x = random.randint(0, 47)
        # y = random.randint(0, 63)
        # score[i] = generate_heatmap(score[i], (x, y), (15, 15))
    # show_heatmap(score)

    # plt.imshow(score, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
