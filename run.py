'''
**************************************************
@File   ：AttitudeRecognition -> run
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/16 21:24
**************************************************
'''
import os
import sys
from math import ceil

import cv2
import torch
import numpy as np
import random

from ultralytics import YOLO

from arguments import init
from cpn.network import cpn
from utils.image_utils import load_image, im_to_torch
from utils.model_utils import get_keypoints, switch
from utils.os_utils import newdir

dirname = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dirname, 'cpn')))
seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
pixel_means = (torch.Tensor([102.9801, 115.9465, 122.7717]) / 255.0).view(1, 3, 1, 1) # BGR
colors = [
    (255, 0, 0),  # blue
    (0, 255, 0),  # green
    (0, 0, 255),  # red
    (255, 255, 0),  # yellow
    (255, 0, 255),  # pink
    (0, 255, 255),  # cyan
    (255, 165, 0),  # orange
    (128, 0, 128),  # purple
    (240, 248, 255)  # Light blue
]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def loader(args):
    model = cpn((64, 48), args).to(args.device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    return model


def predict(model, inputs, inputs_flip):

    global_output, refine_output = model(inputs)
    if inputs_flip is None:
        return get_keypoints(refine_output[0])
    else:
        flip_global_output, flip_refine_output = model(inputs_flip)
        return get_keypoints(refine_output[0], flip_refine_output[0])


def color_normalize(x):
    x -= pixel_means
    return x


def draw(args, image, keypoints_dict):
    keypoints = keypoints_dict["keypoints"]
    shape = keypoints_dict["shape"]
    x_offset, y_offset = keypoints_dict["offset"]
    ratio = shape[0] / 256
    radius = ceil(5 * ratio)
    thickness = ceil(2 * ratio)

    for i in range(17):
        x, y, v = keypoints[i]
        if v > args.lim:
            cv2.circle(image, (int(x * shape[0] + x_offset), int(y * shape[1] + y_offset)),
                       radius, colors[(i + 1) // 2], thickness)

    for i, j in skeleton:
        x1, y1, v1 = keypoints[i - 1]
        x2, y2, v2 = keypoints[j - 1]
        if v1 > args.lim and v2 > args.lim:
            x1, y1 = int(x1 * shape[0] + x_offset), int(y1 * shape[1] + y_offset)
            x2, y2 = int(x2 * shape[0] + x_offset), int(y2 * shape[1] + y_offset)
            cv2.line(image, (x1, y1), (x2, y2), colors[1], thickness)

    return image


def draw_list(args, image, keypoints_list):
    for keypoints in keypoints_list:
        draw(args, image, keypoints)


def people_pose(args, model, image):
    input = im_to_torch(image)
    input = color_normalize(input.view(1, *input.shape)).to(args.device)
    input_flip = None
    if not args.noflip:
        flip_image = image.copy()
        flip_image = cv2.flip(flip_image, 1)
        input_flip = im_to_torch(flip_image)
        input_flip = color_normalize(input_flip.view(1, *input_flip.shape)).to(args.device)
    keypoints = predict(model, input, input_flip)
    return keypoints


def people_detect(args, image):

    return image

def main(args):
    yolo_model = YOLO('yolov8s.pt')
    model = loader(args)

    cap = cv2.VideoCapture(0 if args.source == '0' else args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.source == '0':
        fps = 8

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.save:
        path = newdir('run')
        if not os.path.exists(path):
            os.makedirs(path)
        video_path = os.path.join(path, f'result.mp4')
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    status = args.interval
    keyp = []
    while True:
        ret, image = cap.read()
        if ret:
            if args.source == '0':
                image = cv2.flip(image, 1)
            status += 1
            if status >= args.interval + 1:
                results = yolo_model(image)
                boxes = results[0].boxes
                keyp = []
                count = 0
                for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                    if cls != 0:
                        continue
                    if conf < 0.7 or count + 1 > args.maxn:
                        break
                    count += 1
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1 = max(0, x1 - 5)
                    y1 = max(0, y1 - 5)
                    x2 = min(image.shape[1], x2 + 5)
                    y2 = min(image.shape[0], y2 + 5)
                    cv2.rectangle(image, (x1, y1), (x2, y2), colors[0], 1)
                    new_image = cv2.resize(image[y1:y2, x1:x2], (192, 256))
                    keypoints = people_pose(args, model, new_image)
                    keyp.append({"keypoints": keypoints, "shape": (x2 - x1, y2 - y1), "offset": (x1, y1)})
                status = 0
            draw_list(args, image, keyp)
            cv2.imshow('KRKyyds', image)
            if args.save:
                video_writer.write(image)
            if cv2.waitKey(25) & 0xFF == 27:
                break
        else:
            print("End of video.")
            break

    cap.release()
    if args.save:
        print(f'Video is saved at {video_path}.')
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = init('run')
    main(args)