'''
**************************************************
@File   ：AttitudeRecognition -> train
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/20 10:41
**************************************************
'''

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
from utils.model_utils import accuracy, get_keypoints_batch, switch, checkpoint
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
    train_dataset = DataSet(args, 'train')
    valid_dataset = DataSet(args, 'valid')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)
    model = cpn((64, 48), args).to(args.device)
    criterion1 = nn.MSELoss().to(args.device)
    criterion2 = nn.MSELoss(reduction='none').to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    return train_loader, valid_loader, model, criterion1, criterion2, optimizer


def train(args, train_loader, model, criterion1, criterion2, optimizer):
    model.train()
    train_epoch_loss = []
    global_epoch_loss = []
    refine_epoch_loss = []
    num_keypoints, acc_keypoints = 0, 0
    for inputs, targets, origin_keypoints in tqdm(train_loader):
        inputs = inputs.to(args.device)
        target15, target11, target9, target7 = targets
        origin_keypoints = origin_keypoints.to(args.device)

        global_outputs, refine_outputs = model(inputs)

        global_loss = 0.
        for global_output, label in zip(global_outputs, targets):
            num_points = global_output.size()[1]
            global_label = label * (origin_keypoints[:, :, 2] > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_loss += criterion1(global_output, global_label.to(args.device)) / 2.0
        refine_loss = criterion2(refine_outputs, target7.to(args.device))
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        zero_loss = (refine_loss * torch.Tensor(origin_keypoints[:, :, 2] < 0.1)).sum() / num_points
        refine_loss *= torch.Tensor(origin_keypoints[:, :, 2] > 0.1)
        refine_loss = ohkm(refine_loss, 8)
        loss = global_loss + refine_loss
        if args.zloss:
            loss += zero_loss

        global_epoch_loss.append(global_loss.item())
        refine_epoch_loss.append(refine_loss.item())
        train_epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        keypoints = get_keypoints_batch(refine_outputs, None)
        acc_keypoints += accuracy(keypoints, to_numpy(origin_keypoints), threshold=args.threshold, lim=args.lim)
        num_keypoints += inputs.shape[0] * 17

    return np.average(global_epoch_loss), np.average(refine_epoch_loss), \
           np.average(train_epoch_loss), acc_keypoints / num_keypoints


def valid(args, valid_loader, model, criterion1, criterion2):
    with torch.no_grad():
        model.eval()
        valid_epoch_loss = []
        global_epoch_loss = []
        refine_epoch_loss = []
        num_keypoints, acc_keypoints = 0, 0
        for inputs, inputs_flip, targets, origin_keypoints in tqdm(valid_loader):
            inputs = inputs.to(args.device)
            target15, target11, target9, target7 = targets
            origin_keypoints = origin_keypoints.to(args.device)

            global_outputs, refine_outputs = model(inputs)
            global_outputs_flip, refine_outputs_flip = None, None
            if not args.noflip:
                inputs_flip = inputs_flip.to(args.device)
                global_outputs_flip, refine_outputs_flip = model(inputs_flip)
                global_outputs_flip = torch.flip(global_outputs_flip, dims=[4])
                refine_outputs_flip = torch.flip(refine_outputs_flip, dims=[3])
                global_outputs += switch(global_outputs_flip)
                refine_outputs += switch(refine_outputs_flip)
                global_outputs /= 2
                refine_outputs /= 2

            global_loss = 0.
            for global_output, label in zip(global_outputs, targets):
                num_points = global_output.size()[1]
                global_label = label * (origin_keypoints[:, :, 2] > 1.1).type(torch.FloatTensor).view(-1, num_points, 1,
                                                                                                     1)
                global_loss += criterion1(global_output, global_label.to(args.device)) / 2.0
            refine_loss = criterion2(refine_outputs, target7.to(args.device))
            refine_loss = refine_loss.mean(dim=3).mean(dim=2)
            zero_loss = (refine_loss * torch.Tensor(origin_keypoints[:, :, 2] < 0.1)).sum() / num_points
            refine_loss *= torch.Tensor(origin_keypoints[:, :, 2] > 0.1)
            refine_loss = ohkm(refine_loss, 8)
            loss = global_loss + refine_loss
            if args.zloss:
                loss += zero_loss

            global_epoch_loss.append(global_loss.item())
            refine_epoch_loss.append(refine_loss.item())
            valid_epoch_loss.append(loss.item())

            keypoints = get_keypoints_batch(refine_outputs, None)
            acc_keypoints += accuracy(keypoints, to_numpy(origin_keypoints), threshold=args.threshold, lim=args.lim)
            num_keypoints += inputs.shape[0] * 17

        return np.average(global_epoch_loss), np.average(refine_epoch_loss), \
               np.average(valid_epoch_loss), acc_keypoints / num_keypoints


def logger(**kwargs):
    train_global_loss = kwargs.get('train_global_loss')
    train_refine_loss = kwargs.get('train_refine_loss')
    train_epochs_loss = kwargs.get('train_epochs_loss')
    valid_global_loss = kwargs.get('valid_global_loss')
    valid_refine_loss = kwargs.get('valid_refine_loss')
    valid_epochs_loss = kwargs.get('valid_epochs_loss')
    train_epochs_acc = kwargs.get('train_epochs_acc')
    valid_epochs_acc = kwargs.get('valid_epochs_acc')
    x1 = kwargs.get('x1')
    x2 = kwargs.get('x2')
    path = kwargs.get('path')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].set_title('epochs loss')
    axes[0, 0].plot(x1, train_epochs_loss, '-o', label="train_epochs_loss")
    axes[0, 0].plot(x2, valid_epochs_loss, '-o', label="valid_epochs_loss")
    axes[0, 0].legend()

    axes[0, 1].set_title('epochs acc')
    axes[0, 1].plot(x1, train_epochs_acc, '-o', label="train_epochs_acc")
    axes[0, 1].plot(x2, valid_epochs_acc, '-o', label="valid_epochs_acc")
    axes[0, 1].legend()

    axes[1, 0].set_title('train global:refine')
    axes[1, 0].plot(x1, train_global_loss, '-o', label="train_global_loss")
    axes[1, 0].plot(x2, train_refine_loss, '-o', label="train_refine_loss")
    axes[1, 0].legend()

    axes[1, 1].set_title('valid global:refine')
    axes[1, 1].plot(x1, valid_global_loss, '-o', label="valid_global_loss")
    axes[1, 1].plot(x2, valid_refine_loss, '-o', label="valid_refine_loss")
    axes[1, 1].legend()

    plt.draw()
    plt.savefig(os.path.join(path, 'train_valid_compare.png'))


def main(args):
    try:
        train_loader, valid_loader, model, criterion1, criterion2, optimizer = loader(args)
    except Exception as e:
        print(e)
        sys.exit(0)

    train_global_loss = []
    train_refine_loss = []
    train_epochs_loss = []
    valid_global_loss = []
    valid_refine_loss = []
    valid_epochs_loss = []
    train_epochs_acc = []
    valid_epochs_acc = []

    x1 = []
    x2 = []
    path = newdir('data')
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_path = os.path.join(path, 'checkpoint')
    os.makedirs(checkpoint_path)

    for epoch in tqdm(range(args.epochs)):
        train_global, train_refine, train_loss, train_acc = train(args, train_loader, model, criterion1,
                                                                            criterion2, optimizer)
        train_global_loss.append(train_global)
        train_refine_loss.append(train_refine)
        train_epochs_loss.append(train_loss)
        train_epochs_acc.append(train_acc)
        x1.append(epoch + 1)

        valid_global, valid_refine, valid_loss, valid_acc = valid(args, valid_loader, model, criterion1, criterion2)
        valid_global_loss.append(valid_global)
        valid_refine_loss.append(valid_refine)
        valid_epochs_loss.append(valid_loss)
        valid_epochs_acc.append(valid_acc)
        x2.append(epoch + 1)

        print(f'epoch {epoch + 1} train global loss:', train_global)
        print(f'epoch {epoch + 1} train refine loss:', train_refine)
        print(f'epoch {epoch + 1} train loss:', train_loss)
        print(f'epoch {epoch + 1} valid global loss:', valid_global)
        print(f'epoch {epoch + 1} valid refine loss:', valid_refine)
        print(f'epoch {epoch + 1} valid loss:', valid_loss)
        print(f'epoch {epoch + 1} train acc :', train_acc * 100, '%')
        print(f'epoch {epoch + 1} valid acc :', valid_acc * 100, '%')

        if not args.nosave:
            checkpoint(model, epoch, checkpoint_path)

    logger(
        train_global_loss=train_global_loss,
        train_refine_loss=train_refine_loss,
        train_epochs_loss=train_epochs_loss,
        valid_global_loss=valid_global_loss,
        valid_refine_loss=valid_refine_loss,
        valid_epochs_loss=valid_epochs_loss,
        train_epochs_acc=train_epochs_acc,
        valid_epochs_acc=valid_epochs_acc,
        x1=x1,
        x2=x2,
        path=path,
    )

    print(f'Running message was saved at {path}.')


if __name__ == '__main__':
    args = init('train')
    main(args)
