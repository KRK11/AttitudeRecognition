'''
**************************************************
@File   ：AttitudeRecognition -> dataset
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/20 10:42
**************************************************
'''
import os.path
import random
from os.path import join

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.image_utils import *
from utils.model_utils import switch


class DataSet(Dataset):
    def __init__(self, args, flag='train', shape=(256, 192)) -> None:
        self.args = args
        self.flag = flag
        self.shape = shape
        self.out_shape = (shape[0] // 4, shape[1] // 4)
        self.num_class = args.num_class
        self.pixel_means = (np.array([102.9801, 115.9465, 122.7717], dtype=np.float32) / 255.0).reshape(3, 1, 1)  # BGR
        assert flag in ['train', 'valid', 'test', 'mytest'], 'not implement'
        if self.flag == 'train':
            self.path = os.path.join(args.dataset, 'train_image')
            self.dtf = pd.read_csv(os.path.join(args.dataset, 'train_label.csv'))
        else:
            self.path = os.path.join(args.dataset, 'valid_image')
            self.dtf = pd.read_csv(os.path.join(args.dataset, 'valid_label.csv'))
        self.len = int(self.dtf.shape[0] * args.ratio / 100)

    def flip(self, image, keypoints):
        image = cv2.flip(image, 1)
        keypoints[:, 0] = 1 - keypoints[:, 0]
        keypoints = switch(keypoints)
        return image, keypoints

    def rotate(self, image, keypoints):
        angle = random.uniform(0, 45)
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

    def __getitem__(self, item):
        image_id = self.dtf.iloc[item, 0]
        key_points = self.dtf.iloc[item, 1:18]
        width, height, real_image_id, x_offset, y_offset = 0, 0, 0, 0, 0
        if self.flag == 'test' or self.flag == 'mytest':
            width, height, real_image_id = self.dtf.iloc[item, 18:21]
            try:
                x_offset, y_offset = self.dtf.iloc[item, 21:23]
            except Exception: pass
        key_points = np.array([np.array(i.strip('[]').split(', ')).astype(np.float32) for i in key_points])
        image_path = join(self.path, f'{image_id}.jpg')
        image = load_image(image_path)
        new_image = cv2.resize(image, (192, 256), cv2.INTER_LANCZOS4)

        if self.flag == 'train' and not self.args.noflip and random.randint(0, 3) == 0: # 0.25
            new_image, key_points = self.flip(new_image, key_points)
        elif self.flag == 'train' and not self.args.norotate and random.randint(0, 15) == 0: # 0.75 * 0.0625
            new_image, key_points = self.rotate(new_image, key_points)

        origin_keypoints = key_points.copy()
        key_points[:, 0] = np.round(key_points[:, 0] * (self.shape[1] / 4)).astype(np.int32)
        key_points[:, 1] = np.round(key_points[:, 1] * (self.shape[0] / 4)).astype(np.int32)

        target15 = generate_label(key_points, self.num_class, (15, 15), self.out_shape)
        target11 = generate_label(key_points, self.num_class, (11, 11), self.out_shape)
        target9 = generate_label(key_points, self.num_class, (9, 9), self.out_shape)
        target7 = generate_label(key_points, self.num_class, (7, 7), self.out_shape)
        targets = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]

        flip_image = new_image.copy()
        input = im_to_torch(new_image)
        input_flip = torch.Tensor([])

        if self.flag != 'train' and not self.args.noflip:
            flip_image = cv2.flip(flip_image, 1)
            input_flip = im_to_torch(flip_image)

        if self.flag == 'train':
            input.mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        input -= self.pixel_means
        if self.flag != 'train' and not self.args.noflip:
            input_flip -= self.pixel_means

        if self.flag == 'test':
            return input, input_flip, width, height, real_image_id, x_offset, y_offset
        elif self.flag == 'mytest':
            return input, input_flip, width, height, image_id
        elif self.flag == 'valid':
            return input, input_flip, targets, origin_keypoints
        elif self.flag == 'train':
            return input, targets, origin_keypoints

    def __len__(self):
        return self.len
