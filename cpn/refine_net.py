'''
**************************************************
@File   ：AttitudeRecognition -> refine_net
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/27 1:24
**************************************************
'''

import torch
import torch.nn as nn
from .resnet.bottle_neck import conv1x1, conv3x3


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            conv1x1(inplanes, planes * 2, stride=stride),
            nn.BatchNorm2d(planes * 2),
        )

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(RefineNet, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade - i - 1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4 * lateral_channel, num_class)
        self.num_class = num_class
        self.out_shape = out_shape

    def _make_layer(self, input_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(BottleNeck(input_channel, 128))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        layers = []
        layers.append(BottleNeck(input_channel, 128))
        layers.append(conv3x3(256, num_class))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        refine_fms = []
        for i in range(4):
            refine_fms.append(self.cascade[i](x[i]))
        out = torch.cat(refine_fms, dim=1)
        out = self.final_predict(out)
        # softmax/div on each heatmap
        # out = out.view(-1, self.num_class, self.out_shape[0] * self.out_shape[1])
        # out = torch.softmax(out.detach().clone(), dim=2)
        # max = torch.max(out, dim=2, keepdim=True)[0]
        # min = torch.min(out, dim=2, keepdim=True)[0]
        # out = torch.div(out, max - min)
        # out = out.view(-1, self.num_class, *self.out_shape)
        return out
