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
import pandas as pd
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
    x = min(x, heatmap.shape[1])
    y = min(y, heatmap.shape[0])
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
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, ::-1])
    plt.colorbar()
    plt.title('Image')
    for i in range(len(keypoints)):
        x, y, _ = keypoints[i]
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        plt.plot(x, y, 'yo')
        plt.axis('off')
        heatmap[y][x] = 1

    heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)
    am = np.amax(heatmap)
    heatmap /= am / 255

    for i in range(len(keypoints)):
        plt.subplot(6, 6, i % 3 + (i // 3) * 6 + 4)
        single_heatmap = np.zeros(image.shape[:2])
        x, y, _ = keypoints[i]
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        single_heatmap = generate_heatmap(single_heatmap, (x, y), (101, 101))
        plt.imshow(single_heatmap, cmap='hot', interpolation='nearest')
        colorbar = plt.colorbar()
        if (i % 3 + (i // 3) * 6 + 4) % 6:
            colorbar.set_ticks([])
        plt.axis('off')

    plt.subplot(6, 6, 36)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    # score = np.zeros((20, 20))
    # score = generate_heatmap(score, (10, 10), (15, 15))
    # print(score[10, 10])
    # print(score)
    image = cv2.imread(r'D:\python\MachineLearning\datasets\coco2017-people\train_image\74.jpg')
    dtf = pd.read_csv(r'D:\python\MachineLearning\datasets\coco2017-people\train_label.csv')
    key_points = dtf.iloc[73, 1:18]
    key_points = np.array([np.array(i.strip('[]').split(', ')).astype(np.float32) for i in key_points])
    score = np.zeros(image.shape[:2])
    show_heatmap(score, image, key_points)

    # plt.imshow(score, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
