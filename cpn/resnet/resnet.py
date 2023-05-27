'''
**************************************************
@File   ：AttitudeRecognition -> resnet
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/20 10:36
**************************************************
'''

import torch
import math
from torch import nn
from .conv import conv1x1


class ResNet(nn.Module):
    # size / 32 / 7
    def __init__(self, block, layers, num_class=17):
        super(ResNet, self).__init__()
        # the first layer changes the channel to 64,
        # and the [h,w] will be change to [(h-1)/stride+1,(w-1)/stride+1] after the first layer.
        self.in_channel = 64
        self.block = block
        self.layers = layers

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # there are four block layers, each layers contains more than one block.
        self.stage1 = self._make_layer(self.block, 64, layers[0], stride=1)
        self.stage2 = self._make_layer(self.block, 128, layers[1], stride=2)
        self.stage3 = self._make_layer(self.block, 256, layers[2], stride=2)
        self.stage4 = self._make_layer(self.block, 512, layers[3], stride=2)

        # Initialization the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, plane, block_num, stride=1):
        block_list = []
        downsample = None

        # if the in_channel isn't equal to the out_channel or the shape isn't equal,
        # downsample will be needed to process the (in_channel, shape) as same as the out_channel
        # so that the in_channel can be added to the out_channel to achieve the resnet struct.
        if stride != 1 or self.in_channel != plane * block.extension:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, plane * block.extension, stride),
                nn.BatchNorm2d(plane * block.extension)
            )

        conv_block = block(self.in_channel, plane, stride, downsample=downsample)

        # the first block's in_channel is different to the block_num-1 th in_channel.
        block_list.append(conv_block)
        # modify the in_channel for the next stage layer.
        self.in_channel = plane * block.extension

        for _ in range(1, block_num):
            block_list.append(block(self.in_channel, plane, stride=1))

        return nn.Sequential(*block_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return [x4, x3, x2, x1]


def resnet50(args):
    from .bottle_neck import BottleNeck
    model = ResNet(BottleNeck, [3, 4, 6, 3], args.num_class)
    if args.resnet:
        print('Initialize with pretrained ResNet50')
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = torch.load(args.resnet)
        for k, v in pretrained_state_dict.items():
            if k in state_dict:
                state_dict[k] = v
        print('successfully load ' + str(len(state_dict.keys())) + ' keys')
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    model = resnet50()
