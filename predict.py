'''
**************************************************
@File   ：AttitudeRecognition -> predict
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/20 10:54
**************************************************
'''
import os
import random
import sys
import time
from math import ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from arguments import init
from cpn.network import cpn
from utils.image_utils import load_image, im_to_torch, to_numpy, display_heatmap
from utils.model_utils import switch

dirname = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dirname, 'cpn')))
seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


def get_keypoints(refine_output, flip_refine_output=None):
    single_map = to_numpy(refine_output)
    if flip_refine_output is not None:
        flip_single_map = to_numpy(flip_refine_output)
        fscore = flip_single_map.transpose((1, 2, 0))
        fscore = cv2.flip(fscore, 1)
        fscore = fscore.transpose((2, 0, 1))
        fscore = switch(fscore)
        single_map += fscore
        single_map /= 2

    r0 = single_map.copy()
    r0 /= (r0.max() - r0.min())
    v_score = np.zeros(17)
    keypoints = []
    heatmap = single_map.copy()
    for p in range(17):
        single_map[p] /= np.amax(single_map[p])
        heatmap[p] /= np.amax(heatmap[p])
        border = 10
        dr = np.zeros((64 + 2 * border, 48 + 2 * border))
        dr[border:-border, border:-border] = single_map[p].copy()
        dr = cv2.GaussianBlur(dr, (21, 21), 0)
        lb = dr.argmax()
        y, x = np.unravel_index(lb, dr.shape)
        dr[y, x] = 0
        lb = dr.argmax()
        py, px = np.unravel_index(lb, dr.shape)
        y -= border
        x -= border
        py -= border + y
        px -= border + x
        ln = (px ** 2 + py ** 2) ** 0.5
        delta = 0.25
        if ln > 1e-3:
            x += delta * px / ln
            y += delta * py / ln
        x = max(0, min(x, 48 - 1))
        y = max(0, min(y, 64 - 1))
        resx = float((4 * x + 2) / 192)
        resy = float((4 * y + 2) / 256)
        v_score[p] = float(r0[p, int(round(y) + 1e-10), int(round(x) + 1e-10)])
        keypoints.append([resx, resy, v_score[p]])

    return np.array(keypoints), heatmap

def pred(args):
    model_path, image_path = args.model, args.image
    model = cpn((64, 48), args).to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = load_image(image_path)
    new_image = cv2.resize(image, (192, 256), cv2.INTER_LANCZOS4)
    input = im_to_torch(new_image)
    pixel_means = (np.array([102.9801, 115.9465, 122.7717], dtype=np.float32) / 255.0).reshape(3, 1, 1)  # BGR
    input -= pixel_means
    input = input.view(1, *input.shape).to(args.device)

    ratio = image.shape[0] / 256
    radius = ceil(5 * ratio)
    thickness = ceil(2 * ratio)

    global_output, refine_output = model(input)

    if not args.noflip:
        flip_image = new_image.copy()
        flip_image = cv2.flip(flip_image, 1)
        input_flip = im_to_torch(flip_image)
        input_flip -= pixel_means
        input_flip = input_flip.view(1, *input_flip.shape).to(args.device)
        flip_global_output, flip_refine_output = model(input_flip)
        keypoints, heatmap = get_keypoints(refine_output[0], flip_refine_output[0])
    else:
        keypoints, heatmap = get_keypoints(refine_output[0])

    image = np.array(image)

    colors = [
        (255, 0, 0),  # 蓝色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 红色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 粉色
        (0, 255, 255),  # 青色
        (255, 165, 0),  # 橙色
        (128, 0, 128),  # 紫色
        (240, 248, 255)  # 淡蓝色
    ]

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    for i in range(17):
        x, y, v = keypoints[i]
        if v > args.lim:
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), radius, colors[(i + 1) // 2], thickness)

    for i, j in skeleton:
        x1, y1, v1 = keypoints[i - 1]
        x2, y2, v2 = keypoints[j - 1]
        if v1 > args.lim and v2 > args.lim:
            x1, y1 = int(x1 * image.shape[1]), int(y1 * image.shape[0])
            x2, y2 = int(x2 * image.shape[1]), int(y2 * image.shape[0])
            cv2.line(image, (x1, y1), (x2, y2), colors[1], thickness)

    plt.imshow(image[:, :, ::-1])
    plt.axis('off')
    plt.show()

    print(heatmap.shape)
    display_heatmap(heatmap, image, keypoints)


if __name__ == '__main__':
    args = init('predict')
    pred(args)
