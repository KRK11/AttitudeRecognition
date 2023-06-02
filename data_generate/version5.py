'''
**************************************************
@File   ：AttitudeRecognition -> version5
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/30 13:41
**************************************************
'''

import json
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np


def generate_image_label(mode):
    if mode == 'train':
        path = '/kaggle/input/coco-2017-dataset/coco2017/annotations/person_keypoints_train2017.json'
        save_path = '/kaggle/working/train_image'
        image_dir = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
        label_path = 'train_label.csv'
    else:
        path = '/kaggle/input/coco-2017-dataset/coco2017/annotations/person_keypoints_val2017.json'
        save_path = '/kaggle/working/valid_image'
        image_dir = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
        label_path = 'valid_label.csv'

    count = 0

    columns = ['image_id']
    columns.extend([f'keypoint{i}' for i in range(1, 18)])
    columns.extend(['width', 'height', 'real_image_id', 'x_offset', 'y_offset'])
    label = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(path, 'r') as f:
        data = json.load(f)
    mean_value = np.array([102.9801, 115.9465, 122.7717])  # BGR
    for single_data in tqdm(data['annotations']):
        try:
            image_id = single_data['image_id']
            image_path = os.path.join(image_dir, str(image_id).zfill(12) + '.jpg')
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # extend the image
            add = max(image.shape[0], image.shape[1])
            bimg = cv2.copyMakeBorder(image, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                                      value=mean_value.tolist())
            key_points = single_data['keypoints']
            x, y, w, h = single_data['bbox']
            x1, y1, x2, y2 = int(x + add), int(y + add), int(x + w + add), int(y + h + add)
            objcenter = ((x1 + x2) // 2, (y1 + y2) // 2)
            keypoints = np.array(key_points).reshape(17, 3).astype(np.float32)
            keypoints[:, :2] += add
            crop_width, crop_height = (x2 - x1) * 1.2, (y2 - y1) * 1.3

            if crop_height / height > crop_width / width:
                crop_size = crop_height
                min_shape = height
            else:
                crop_size = crop_width
                min_shape = width

            crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
            crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
            crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
            crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

            min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
            max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
            min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
            max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

            keypoints[:, 0] -= min_x
            keypoints[:, 1] -= min_y
            keypoints[:, 0] /= (max_x - min_x)
            keypoints[:, 1] /= (max_y - min_y)

            new_image = bimg[min_y:max_y, min_x:max_x]
            count += 1
            temp = [count]

            for x, y, v in keypoints:
                if x < 0 or x >= 1 or y < 0 or y >= 1:
                    v = 0
                temp.append([x, y, v])
            temp.extend([max_x - min_x, max_y - min_y, image_id, min_x - add, min_y - add])
            cv2.imwrite(os.path.join(save_path, f'{count}.jpg'), new_image)
            label.append(temp)
        except Exception as e:
            print(e)
    dtf = pd.DataFrame(columns=columns, data=label)
    dtf.to_csv(label_path, index=False)


generate_image_label('train')
generate_image_label('valid')