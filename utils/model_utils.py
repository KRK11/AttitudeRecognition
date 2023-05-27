'''
**************************************************
@File   ：AttitudeRecognition -> model_utils
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/28 15:34
**************************************************
'''
import os.path
import torch
import cv2
import numpy as np
from utils.image_utils import to_numpy


def switch(tensor):
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    order = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    if len(tensor.shape) >= 5:  # stage, batch, num_class, H, W
        tensor[:, :] = tensor[:, :, order]
    elif len(tensor.shape) >= 4:  # batch, num_class, H, W
        tensor[:, ] = tensor[:, order]
    else:  # num_class, H, W / num_class, 3
        tensor = tensor[order]
    return tensor


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
    for p in range(17):
        single_map[p] /= np.amax(single_map[p])
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

    return np.array(keypoints)


def get_keypoints_batch(refine_outputs, flip_refine_outputs):
    keypoints = []
    for i in range(refine_outputs.shape[0]):
        flip = flip_refine_outputs[i] if flip_refine_outputs is not None else None
        keypoints.append(get_keypoints(refine_outputs[i], flip))
    return np.array(keypoints)


def checkpoint(model, epoch, path):
    model_path = os.path.join(path, f'{epoch}.pth')
    torch.save(model.state_dict(), model_path)


# all the keypoints are graphic percentage.
def accuracy(label_keypoints, predict_keypoints, threshold=0.1, lim=0.):
    count = np.sum(np.logical_and(label_keypoints[:, :, 2] < lim, predict_keypoints[:, :, 2] < lim))
    distance = np.sqrt(np.sum((label_keypoints[:, :, :2] - predict_keypoints[:, :, :2]) ** 2, axis=2))
    logical = np.logical_and(label_keypoints[:, :, 2] >= lim, predict_keypoints[:, :, 2] >= lim)
    count += np.sum(np.logical_and(logical, distance <= threshold))
    return count


if __name__ == '__main__':
    a = np.random.randint(0, 64, size=(3, 17, 3))
    b = np.random.randint(0, 64, size=(3, 17, 3))
    a[:, :, 2] = np.random.randint(0, 2, size=17)
    b[:, :, 2] = np.random.randint(0, 2, size=17)
    print(a)
    print(b)
    print(accuracy(a, b, 50))
    # final_outputs = torch.randn(1, 2, 2, 2, 2)
    # keypoints = get_keypoints(final_outputs)
    # print(keypoints)
    # a = torch.randn(17, 1, 1)
    # print(a)
    # b = switch(a)
    # print(b)
    # print(a == b)