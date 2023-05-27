'''
**************************************************
@File   ：AttitudeRecognition -> main
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/20 10:03
**************************************************
'''

import json
import os.path
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from cpn.refine_net import BottleNeck
from cpn.resnet.conv import conv1x1, conv3x3
from utils.image_utils import generate_heatmap
from utils.model_utils import accuracy, switch


def rotate(image, keypoints):
    angle = random.uniform(0, 15)
    if random.randint(0, 1):
        angle *= -1
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotMat, (width, height))
    for i in range(17):
        x, y, v = keypoints[i]
        coor = np.array([x * width, y * height, 1])
        if v > 0:
            coor = np.dot(rotMat, coor)
        v *= ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height))
        keypoints[i] = (coor[0] / width, coor[1] / height, v)
    return image, keypoints


if __name__ == '__main__':

    a = np.random.randn(2, 3, 1, 1)
    print(a)
    order = [0, 2, 1]
    a[:] = a[:, order]
    print(a)
    # image = cv2.imread(r'D:\python\MachineLearning\datasets\coco2017-people\train_image\33.jpg')
    # dtf = pd.read_csv(r'D:\python\MachineLearning\datasets\coco2017-people\train_label.csv')
    # keypoints = dtf.iloc[32, 1:18]
    # width, height = dtf.iloc[32, 18:20]
    # keypoints = np.array([np.array(i.strip('[]').split(', ')).astype(np.float32) for i in keypoints])
    # new_image = image.copy()
    #
    # for x, y, v in keypoints:
    #     if v > 0:
    #         cv2.circle(image, (int(x * width), int(y * height)), 4, (255, 0, 255), -1)
    #
    # cv2.imshow('result1', image)
    # cv2.waitKey(0)
    # center = (image.shape[1] // 2, image.shape[0] // 2)
    #
    # new_image, keypoints = rotate(new_image, keypoints)
    # for x, y, v in keypoints:
    #     if v > 0:
    #         cv2.circle(new_image, (int(x * width), int(y * height)), 4, (255, 0, 255), -1)
    #
    # cv2.imshow('result2', new_image)
    # cv2.waitKey(0)

    # with open('test/2023-05-14_16-26-09/result.json', 'r') as f:
    #     data = json.load(f)
    # image_id = data[25]['image_id']
    # keypoints = data[25]['keypoints']
    #
    # image = cv2.imread(fr'D:\python\MachineLearning\datasets\coco2017-people\valid_image/{image_id}.jpg')
    # for i in range(0, 51, 3):
    #     x, y, v = keypoints[i:i+3]
    #     cv2.circle(image, (x, y), 4, (255, 0, 255), -1)
    # plt.imshow(image[:, :, ::-1])
    # plt.show()
    # dtf = pd.read_csv('D:\python\MachineLearning\datasets\coco2017-people/train_label.csv')
    # dtf = dtf.assign(real_image_id=dtf['image_id'], x_offset=0, y_offset=0)
    # dtf.to_csv('D:\python\MachineLearning\datasets\coco2017-people/train_label.csv', index=False)
    #
    # dtf = pd.read_csv('D:\python\MachineLearning\datasets\coco2017-people/valid_label.csv')
    # dtf = dtf.assign(real_image_id=dtf['image_id'], x_offset=0, y_offset=0)
    # dtf.to_csv('D:\python\MachineLearning\datasets\coco2017-people/valid_label.csv', index=False)

    # heatmap = np.zeros((8, 8))
    # heatmap = generate_heatmap(heatmap, (4, 4), (5, 5))
    # heatmap /= heatmap.max() - heatmap.min()
    # print(heatmap)
    # sys.exit(0)

    # a = torch.randn((1, 2, 2, 2))
    # print(a)
    # layers = []
    # layers.append(BottleNeck(2, 2))
    # layers.append(conv3x3(4, 17))
    # layers.append(nn.BatchNorm2d(17))
    # b = nn.Sequential(*layers)
    # c = b(a)
    # print(c)
    # print(a)
    # sys.exit(0)
    # a = torch.randn((1, 2, 2, 2))
    # print(a)
    # a = torch.flip(a, dims=[3])
    # print(a)
    sys.exit(0)

    # a = torch.tensor([[1, 2, 3, 4], [4, 6, 7, 8]], dtype=torch.float32, requires_grad=True)

    # b = torch.max(a, dim=1, keepdim=True)[0] - torch.min(a, dim=1, keepdim=True)[0]
    # print(b)
    # a = torch.div(a, b)
    # loss = a.sum()
    # loss.backward()

    # a = torch.tensor([[[1, 2], [3, 1]], [[1, 4], [3, 2]]]).float()
    # a = np.random.randn(2, 2, 2, 2)
    # print(a)
    # a = cv2.flip(a, 1)
    # print(a)

    # a = a.view(2, -1)
    # a = torch.softmax(a, dim=1)
    # a = a.view(2, 2, 2)
    # print(a)
    # with open(
    #         r'D:/python/MachineLearning/datasets/coco2017/person_keypoints_val2017.json/person_keypoints_val2017.json',
    #         'r') as f:
    #     data = json.load(f)
    #
    # print(data.keys())
    # print(data['categories'])
    # print(len(data['categories'][0]['keypoints']))
    # for i in data['annotations']:
    #     print(i.keys())
    #     break
    #
    # for i in data['annotations']:
    #     print(i['keypoints'])
    #     break
    #
    # print(len(data['annotations']))
    #
    # try:
    #     pass
    # except Exception as e:
    #     print(e)
    # train_epochs_acc = [1, 2, 3, 4, 5, 6]
    # valid_epochs_acc = [1, 4, 7]
    # plt.figure(figsize=(12, 4))
    # plt.plot([1, 2, 3, 4, 5, 6], train_epochs_acc, '-o', label="train_epochs_acc")
    # plt.plot([2, 4, 6], valid_epochs_acc, '-o', label="valid_epochs_acc")
    # plt.title("epochs_acc")
    # plt.legend()
    # plt.show()

    path = r'D:/python/MachineLearning/datasets/coco2017-people'
    image_id = 51
    dtf = pd.read_csv(os.path.join(path, 'train_label.csv'))
    image = cv2.imread(os.path.join(path, f'train_image/{image_id}.jpg'), cv2.IMREAD_COLOR)

    key_points = dtf.iloc[image_id - 1, 1:18]
    key_points = np.array([np.array(i.strip('[]').split(', ')).astype(np.float32) for i in key_points])
    width, height = dtf.iloc[image_id - 1, 18:20]

    new_image = image.copy()

    # for i in range(17):
    #     x, y, v = key_points[i]
    #     if v > 0:
    #         cv2.circle(image, (int(x * width), int(y * height)), 3, (63, 255, 127), 3)

    pred = [
            [0.65625, 0.10546875, 0.8189419507980347] ,
            [0.6614583333333334, 0.0859375, 0.8503710031509399] ,
            [0.609375, 0.0859375, 0.97826087474823] ,
            [0.6770833333333334, 0.08984375, 0.7512538433074951] ,
            [0.4635416666666667, 0.0859375, 0.8588054776191711] ,
            [0.6979166666666666, 0.19921875, 0.8211130499839783] ,
            [0.28125, 0.20703125, 0.7798421382904053] ,
            [0.8020833333333334, 0.33984375, 0.8284858465194702] ,
            [0.13541666666666666, 0.35546875, 0.8485900163650513] ,
            [0.90625, 0.47265625, 0.9339712858200073] ,
            [0.08854166666666667, 0.4765625, 0.8825643062591553] ,
            [0.6822916666666666, 0.4921875, 0.814569890499115] ,
            [0.390625, 0.4921875, 0.888455867767334] ,
            [0.765625, 0.7109375, 0.6366897225379944] ,
            [0.421875, 0.7265625, 0.8487075567245483] ,
            [0.4895833333333333, 0.82421875, 0.6995452642440796] ,
            [0.46875, 0.94140625, 0.853518009185791] ,
            ]

    pred = np.array(pred)

    # for i in range(17):
    #     x, y, v = pred[i]
    #     if v > 0:
    #         cv2.circle(new_image, (int(x * width), int(y * height)), 3, (63, 255, 127), 3)
    #
    # cv2.imshow('apple', image)
    # cv2.imshow('banana', new_image)
    # cv2.waitKey(0)

    for i in range(17):
        x1, y1, v1 = key_points[i]
        x2, y2, v2 = pred[i]
        print(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
        image1 = image.copy()
        cv2.circle(image1, (int(x1 * width), int(y1 * height)), width // 10, (63, 255, 127), 3)
        cv2.circle(image1, (int(x2 * width), int(y2 * height)), 3, (127, 63, 255), 3)
        # cv2.imshow('1', image1)
        # cv2.waitKey(0)

    key_points = np.expand_dims(key_points, axis=0)
    pred = np.expand_dims(pred, axis=0)
    print(key_points)
    print(pred)
    print(accuracy(key_points, pred))

    # {
    #     "supercategory": "person",
    #     "id": 1,
    #     "name": "person",
    #     "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    #                   "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    #                   "right_knee", "left_ankle", "right_ankle"],
    #     "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    #                  [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    # }

    angle = random.uniform(0, 45)
    if random.randint(0, 1):
        angle *= -1
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotMat, (width, height))
    for i in range(17):
        x, y, v = key_points[0][i]
        coor = np.array([x * width, y * height, 1])
        if x >= 0 and y >= 0:
            coor = np.dot(rotMat, coor)
        v *= ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height))
        key_points[0][i] = (*coor[:2], v)
        print(key_points[0][i])
        if v > 0:
            cv2.circle(image, (int(coor[0]), int(coor[1])), 5, (255, 0, 255), -1)

    cv2.imshow('result', image)
    cv2.waitKey(0)