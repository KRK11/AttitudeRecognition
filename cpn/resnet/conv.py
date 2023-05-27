'''
**************************************************
@File   ：AttitudeRecognition -> conv
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/20 10:05
**************************************************
'''
from torch import nn


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)