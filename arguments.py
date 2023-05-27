'''
**************************************************
@File   ：AttitudeRecognition -> arguments
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/20 16:55
**************************************************
'''
import argparse
import torch


def init(mode):
    desc = {'train': 'AttitudeRecognition Train.',
            'predict': 'AttitudeRecognition Predict.',
            'test': 'AttitudeRecognition Test.',
            'run': 'AttitudeRecognition Run.'}
    parser = argparse.ArgumentParser(description=desc[mode])
    parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu",
                        type=str, metavar='cuda:0/cpu',
                        help=f'run on single gpu or cpu (default: {"cuda:0" if torch.cuda.is_available() else "cpu"})')
    parser.add_argument('--model', default=None, type=str, metavar='PATH',
                        help='pretrain model for train or model for predict (default: None)')
    parser.add_argument('--num_class', default=17, type=int, metavar='N',
                        help='number of the key points (default: 17)')
    parser.add_argument('--noflip', action='store_true',
                        help='train: not random flip| others: final result is not calculated with the flip image')
    parser.add_argument('--lim', default=0.18, type=float, metavar='N',
                        help='the conf bound of result (default: 0.18)')
    parser.add_argument('--resnet', default=None, type=str, metavar='PATH',
                        help='initial the resnet50 (default: None)')
    if mode == 'train':
        parser.add_argument('--norotate', action='store_true',
                            help='not random rotate')
        parser.add_argument('--epochs', default=10, type=int, metavar='N',
                            help='number of total epochs to run (default: 10)')
        parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                            help='batch size to run (default: 2)')
        parser.add_argument('--lr', default=0.001, type=float, metavar='N',
                            help='learning rate of the network (default: 0.001)')
        parser.add_argument('--dataset', default='D:/python/MachineLearning/datasets/coco2017-people',
                            type=str, metavar='Dirname', help='the dirname of dataset (default: None)')
        parser.add_argument('--ratio', default=100., type=float, metavar='N',
                            help='the ratio(percentage 0-100) of the dataset (default: 100.0)')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save the model')
        parser.add_argument('--threshold', default=0.1, type=float, metavar='N',
                            help='the accuracy of different distance threshold(0-1) (default: 0.1)')
    elif mode == 'predict':
        parser.add_argument('--image', default=None, type=str, metavar='PATH',
                            help='the image for predict (default: None)')
    elif mode == 'test':
        parser.add_argument('--dataset', default='D:/python/MachineLearning/datasets/coco2017-people',
                            type=str, metavar='Dirname', help='the dirname of dataset (default: None)')
        parser.add_argument('--ratio', default=100., type=float, metavar='N',
                            help='the ratio(percentage 0-100) of the dataset (default: 100.0)')
        parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                            help='batch size to run (default: 2)')
    elif mode == 'run':
        parser.add_argument('--source', default='0', type=str, metavar='PATH',
                            help='camera: 0 | the path of the video) (default: 0)')
        parser.add_argument('--interval', default=0, type=int, metavar='N',
                            help='the interval between two processed frame (default: 0)')
        parser.add_argument('--maxn', default=2, type=int, metavar='N',
                            help='the max number of detected people (default: 2)')
    return parser.parse_args()
