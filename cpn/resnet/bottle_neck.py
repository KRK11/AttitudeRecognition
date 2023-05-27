'''
**************************************************
@File   ：AttitudeRecognition -> Bottleneck
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/20 10:04
**************************************************
'''
from torch import nn
from .conv import conv1x1, conv3x3


class BottleNeck(nn.Module):
    extension = 4

    # Bottleneck only decrease the [h,w] in conv1 when stride > 1,
    # so the [h,w] is to be [(h-1)/stride+1,(w-1)/stride+1].
    # the in_channel will be change to channel*extension.
    # channel is the temp variable.
    # bottleneck 1x1 3x3 1x1
    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()

        self.conv1 = conv1x1(in_channel, channel, stride)
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = conv3x3(channel, channel)
        self.bn2 = nn.BatchNorm2d(channel)

        self.conv3 = conv1x1(channel, channel * self.extension)
        self.bn3 = nn.BatchNorm2d(channel * self.extension)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
