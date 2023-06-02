'''
**************************************************
@File   ：AttitudeRecognition -> version1
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/30 13:40
**************************************************
'''

from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = YOLO('yolov8x-pose-p6.pt')
columns = ['image_id']
columns.extend([f'keypoint{i}' for i in range(1, 18)])
columns.extend(['width', 'height'])


def keypoints_generate(path, save_path):
    data = []
    error_log = []
    for image_path in os.listdir(path):
        try:
            results = model(os.path.join(path, image_path))

            boxes = results[0].boxes
            h, w = boxes.orig_shape.cpu().numpy()
            keypoints = results[0].keypoints.cpu().numpy().copy()

            keypoints[:, :, 0] /= w
            keypoints[:, :, 1] /= h
            keypoints[:, :, 2] = np.round(keypoints[:, :, 2])
            keypoints = keypoints.tolist()

            max_index = results[0].boxes.data[:, 4].max(axis=0)[1]
            h, w = results[0].plot().shape[:2]
            temp = [int(image_path[:-4])]
            temp.extend(keypoints[max_index])
            temp.extend([w, h])
            data.append(temp)
        except Exception as e:
            error_log.append(image_path)

    data.sort(key=lambda x: x[0])
    dtf = pd.DataFrame(columns=columns, data=data)
    dtf.to_csv(save_path, index=False)
    error_log.sort()
    dtf = pd.DataFrame(columns=['image_id'], data=error_log)
    dtf.to_csv(save_path[:-4] + '_log.csv', index=False)


keypoints_generate('/kaggle/input/coco2017-singlepeople-detect-datasetyolov8/people-dataset/train',
                   'keypoints_label_train.csv')
keypoints_generate('/kaggle/input/coco2017-singlepeople-detect-datasetyolov8/people-dataset/valid',
                   'keypoints_label_valid.csv')