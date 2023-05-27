'''
**************************************************
@File   ：AttitudeRecognition -> global_net
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/27 1:23
**************************************************
'''

import torch.nn as nn
import torch
import math
from .resnet.conv import conv1x1, conv3x3


class GlobalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(GlobalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(conv1x1(input_size, 256))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(conv1x1(256, 256))
        layers.append(nn.BatchNorm2d(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(conv1x1(256, 256))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(conv3x3(256, num_class))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms = []
        global_outs = []
        # [2048, 1024, 512, 256]
        up = None
        for i in range(len(self.channel_settings)):
            if i == 0:
                # channel is changed to 256
                feature = self.laterals[i](x[i])
            else:
                # add the previous feature
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            # upsample the feature to match the next feature
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)
            # fms change the channel to 256 only，
            # outs shape is output_shape(64, 48),channel is num_class,
            # and there are four outputs corresponding to four gauss heatmap.
        global_outs = torch.stack(global_outs, dim=0)
        return global_fms, global_outs
