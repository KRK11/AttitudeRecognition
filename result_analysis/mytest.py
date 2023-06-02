'''
**************************************************
@File   ：AttitudeRecognition -> mytest.py
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/31 1:56
**************************************************
'''

import json
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)
print(os.path.abspath(dirname))
sys.path.append(os.path.abspath(dirname))
sys.path.append(os.path.abspath(os.path.join(dirname, 'cpn')))

from arguments import init
from cpn.network import cpn
from dataset import DataSet
from utils.image_utils import to_numpy
from utils.model_utils import accuracy, get_keypoints_batch, switch
from utils.os_utils import newdir

seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


# torch.autograd.set_detect_anomaly(True)


def ohkm(loss, top_k):
    ohkm_loss = 0.
    for i in range(loss.size()[0]):
        sub_loss = loss[i]
        topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
        tmp_loss = torch.gather(sub_loss, 0, topk_idx)
        ohkm_loss += torch.sum(tmp_loss) / top_k
    ohkm_loss /= loss.size()[0]
    return ohkm_loss


def loader(args):
    test_dataset = DataSet(args, 'mytest')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    model = cpn((64, 48), args).to(args.device)
    return test_loader, model


def test(args, test_loader, model):
    with torch.no_grad():
        model.eval()
        result = []
        for inputs, inputs_flip, width, height, image_id in tqdm(test_loader):
            inputs = inputs.to(args.device)
            global_outputs, refine_outputs = model(inputs)
            global_outputs_flip, refine_outputs_flip = None, None
            if not args.noflip:
                inputs_flip = inputs_flip.to(args.device)
                global_outputs_flip, refine_outputs_flip = model(inputs_flip)

            keypoints = get_keypoints_batch(refine_outputs, refine_outputs_flip)
            for i in range(inputs.shape[0]):
                single_keypoints = keypoints[i]
                single_width = width[i]
                single_height = height[i]
                single_image_id = image_id[i]
                single_result = [single_image_id.item()]
                for x, y, v in single_keypoints:
                    single_result.append([x, y, v])
                single_result.extend([single_width.item(), single_height.item()])
                result.append(single_result)

    return result


def main(args):
    try:
        test_loader, model = loader(args)
    except Exception as e:
        print(e)
        sys.exit(0)

    result = test(args, test_loader, model)

    path = newdir('test')
    if not os.path.exists(path):
        os.makedirs(path)
    result_path = os.path.join(path, 'result.csv')

    columns = ['image_id']
    columns.extend([f'keypoint{i}' for i in range(1, 18)])
    columns.extend(['width', 'height'])
    result.sort(key=lambda x: int(x[0]))

    dtf = pd.DataFrame(columns=columns, data=result)
    dtf.to_csv(result_path, index=False)
    print(f'save the result csv at {result_path}.')


if __name__ == '__main__':
    args = init('test')
    main(args)


