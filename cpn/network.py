'''
**************************************************
@File   ：AttitudeRecognition -> network
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/27 1:49
**************************************************
'''
import torch
import torch.nn as nn
from .resnet.resnet import resnet50
from .global_net import GlobalNet
from .refine_net import RefineNet

__all__ = ['cpn']


class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.global_net = GlobalNet(channel_settings, output_shape, num_class)
        self.refine_net = RefineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out


def cpn(out_size, args):
    res50 = resnet50(args)
    model = CPN(res50, output_shape=out_size, num_class=args.num_class)
    if args.model:
        model.load_state_dict(torch.load(args.model))
        print('Successfully load the pretrained model.')
    return model
