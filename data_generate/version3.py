'''
**************************************************
@File   ：AttitudeRecognition -> version3
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/30 13:40
**************************************************
'''

import json
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm


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
    for single_data in tqdm(data['annotations']):
        try:
            image_id = single_data['image_id']
            #             print(image_id)
            image_path = os.path.join(image_dir, str(image_id).zfill(12) + '.jpg')
            image = cv2.imread(image_path)
            key_points = single_data['keypoints']
            x, y, w, h = single_data['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            for i in range(0, len(key_points), 3):
                x, y, v = key_points[i:i + 3]
                if v > 0:
                    x1 = max(min(x1, x - 10), 0)
                    y1 = max(min(y1, y - 10), 0)
                    x2 = min(max(x2, x + 10), image.shape[1])
                    y2 = min(max(y2, y + 10), image.shape[0])
            new_image = image[y1:y2, x1:x2]
            count += 1
            temp = [count]

            for i in range(0, len(key_points), 3):
                x, y, v = key_points[i:i + 3]
                if v > 0:
                    x, y = (x - x1) / (x2 - x1), (y - y1) / (y2 - y1)
                else:
                    x, y = -10000, -10000
                temp.append([x, y, v])
            temp.extend([x2 - x1, y2 - y1, image_id, x1, y1])
            cv2.imwrite(os.path.join(save_path, f'{count}.jpg'), new_image)
            label.append(temp)
        except Exception as e:
            print(e)
    #         break
    dtf = pd.DataFrame(columns=columns, data=label)
    dtf.to_csv(label_path, index=False)


generate_image_label('train')
generate_image_label('valid')