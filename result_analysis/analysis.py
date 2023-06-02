'''
**************************************************
@File   ：AttitudeRecognition -> analysic
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/31 0:20
**************************************************
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def pretrain(row):
    names = row.columns
    x = np.arange(1, 41)
    print(names)
    plt.rcParams['font.size'] = 6
    plt.subplot(2, 2, 1)
    plt.title('train loss')
    plt.plot(x, row.iloc[:, 1], 'b-', label='global')
    plt.plot(x, row.iloc[:, 2], 'c-', label='refine')
    plt.plot(x, row.iloc[:, 3], 'g-', label='loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.title('valid loss')
    plt.plot(x, row.iloc[:, 4], 'b-', label='global')
    plt.plot(x, row.iloc[:, 5], 'c-', label='refine')
    plt.plot(x, row.iloc[:, 6], 'g-', label='loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.title('train:valid loss')
    plt.plot(x, row.iloc[:, 3], label='train')
    plt.plot(x, row.iloc[:, 6], label='valid')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.title('train:valid accuracy')
    plt.plot(x, row.iloc[:, 7], label='train')
    plt.plot(x, row.iloc[:, 8], label='valid')
    plt.legend()
    plt.grid(True)
    plt.show()


def draw(row, start, end):
    names = row.columns
    x = np.arange(start, end)
    print(names)
    plt.rcParams['font.size'] = 6
    plt.subplot(2, 3, 1)
    plt.title('train loss')
    plt.plot(x, row.iloc[:, 1], 'y-', label='global')
    plt.plot(x, row.iloc[:, 2], 'c-', label='refine')
    plt.plot(x, row.iloc[:, 3], 'g-', label='loss')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.subplot(2, 3, 2)
    plt.title('valid loss')
    plt.plot(x, row.iloc[:, 4], 'y-', label='global')
    plt.plot(x, row.iloc[:, 5], 'c-', label='refine')
    plt.plot(x, row.iloc[:, 6], 'g-', label='loss')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.subplot(2, 3, 3)
    plt.title('train:valid loss')
    plt.plot(x, row.iloc[:, 3], label='train')
    plt.plot(x, row.iloc[:, 6], label='valid')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.subplot(2, 3, 4)
    plt.title('train:valid accuracy')
    plt.plot(x, row.iloc[:, 7], label='train')
    plt.plot(x, row.iloc[:, 8], label='valid')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.subplot(2, 3, 5)
    plt.title('AP')
    plt.plot(x, row.iloc[:, 9], label='mAP')
    plt.plot(x, row.iloc[:, 10], label='mAP-medium')
    plt.plot(x, row.iloc[:, 11], label='mAP-large')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.subplot(2, 3, 6)
    plt.title('vaild')
    plt.plot(x, row.iloc[:, 6] / 200, label='loss/200')
    plt.plot(x, row.iloc[:, 8], label='acc')
    plt.plot(x, row.iloc[:, 9], label='mAP')
    plt.legend()
    plt.grid(True)
    plt.show()


def mydataset(path):
    dtf = pd.read_csv(path)
    pretrain(dtf.iloc[0:40, :])
    # draw(dtf.iloc[40:58, :], 41, 59)
    # dtf = dtf.iloc[40:58, :].append(dtf.iloc[76:, :])
    # draw(dtf, 41, 74)


def original_dataset(path):
    dtf = pd.read_csv(path)
    draw(dtf, 1, 36)


def main():
    # mydataset('../test/DataSet/MyDataSet.csv')
    original_dataset('../test/DataSet/OriginalDataSet.csv')


if __name__ == '__main__':
    main()