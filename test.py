'''
**************************************************
@File   ：AttitudeRecognition -> test
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/14 2:52
**************************************************
'''
import json
import os
import random
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from arguments import init
from cpn.network import cpn
from dataset import DataSet
from utils.image_utils import to_numpy
from utils.model_utils import accuracy, get_keypoints_batch, switch
from utils.os_utils import newdir

dirname = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dirname, 'cpn')))
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
    test_dataset = DataSet(args, 'test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    model = cpn((64, 48), args).to(args.device)
    return test_loader, model


def test(args, test_loader, model):
    with torch.no_grad():
        model.eval()
        result = []
        for inputs, inputs_flip, width, height, real_image_id, x_offset, y_offset in tqdm(test_loader):
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
                single_image_id = real_image_id[i]
                single_x_offset = x_offset[i]
                single_y_offset = y_offset[i]
                single_keypoints[:, 0] *= single_width.item()
                single_keypoints[:, 1] *= single_height.item()
                single_result = []
                for x, y, v in single_keypoints:
                    single_result.extend([round(x + single_x_offset.item()), round(y + single_y_offset.item()), 1])
                single_result_dict = dict()
                single_result_dict['image_id'] = int(single_image_id)
                single_result_dict['category_id'] = 1
                single_result_dict['keypoints'] = single_result
                single_result_dict['score'] = np.average(single_keypoints[:, 2])
                result.append(single_result_dict)

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
    result_path = os.path.join(path, 'result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f)
    print(f'save the result json at {result_path}.')


if __name__ == '__main__':
    args = init('test')
    main(args)
