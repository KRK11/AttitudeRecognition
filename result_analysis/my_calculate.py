'''
**************************************************
@File   ：AttitudeRecognition -> my_calculate
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/31 21:40
**************************************************
'''
import datetime
import os.path
import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dir = os.path.dirname(__file__)
dir = os.path.dirname(dir)
sys.path.append(os.path.abspath(dir))


def nonexistent(x, y, lim):
    return np.sum(np.logical_and(x[:, :, 2] < lim, y[:, :, 2] < lim))


def existent(x, y, threshold, lim):
    exist = np.logical_and(x[:, :, 2] >= lim, y[:, :, 2] >= lim)
    distance = ((x[:, :, 0] - y[:, :, 0]) ** 2 + (x[:, :, 1] - y[:, :, 1]) ** 2) ** 0.5
    return np.sum(np.logical_and(exist, distance <= threshold))


def main(path, save_path):
    dt = pd.read_csv(path)
    gt = pd.read_csv('../test/DataSet/label.csv')

    result = []
    a = dt.iloc[:, 1:18]
    keypoints_dt = np.array([[
        np.array(j.strip('[]').split(', ')) for j in a.iloc[i]] for i in range(a.shape[0])]).astype(np.float32)
    b = gt.iloc[:, 1:18]
    keypoints_gt = np.array([[
        np.array(j.strip('[]').split(', ')) for j in b.iloc[i]] for i in range(b.shape[0])]).astype(np.float32)

    for i in tqdm(np.arange(0, 1.001, 0.001)):
        exist = existent(keypoints_dt, keypoints_gt, 0.1, i) / dt.shape[0] / 17
        noexist = nonexistent(keypoints_dt, keypoints_gt, i) / dt.shape[0] / 17
        result.append([exist, noexist, exist + noexist, i])

    dtf = pd.DataFrame(data=result, columns=['existent', 'nonexistent', 'average', 'lim'])
    dtf.to_csv(save_path, index=False)


def analysis(alpha=0.8):
    dt_highest = pd.read_csv('../test/DataSet/9-3.csv')
    dt_zloss = pd.read_csv('../test/DataSet/18-4.csv')
    best_highest = dt_highest.iloc[:, 0] * alpha + dt_highest.iloc[:, 1] * (1 - alpha)
    best_zloss = dt_zloss.iloc[:, 0] * alpha + dt_zloss.iloc[:, 1] * (1 - alpha)

    x = dt_highest.iloc[:, 3]
    plt.plot(x, dt_highest.iloc[:, 0], label='existent')
    plt.plot(x, dt_highest.iloc[:, 1], label='nonexistent')
    plt.plot(x, dt_highest.iloc[:, 2], label='all')
    plt.plot(x, best_highest, label=f'alpha:{alpha:.2f}')

    ymax1 = max(dt_highest.iloc[:, 2])
    xpos1 = dt_highest.iloc[:, 2].tolist().index(ymax1)
    xmax1 = x[xpos1]
    plt.text(xmax1, ymax1, f'(x:{xmax1:.3f}, y:{ymax1:.3f}, ex:{dt_highest.iloc[xpos1, 0]:.3f})', ha='left',
             va='bottom', fontsize=12)
    plt.plot(xmax1, ymax1, 'ro')
    plt.title('Accuracy highest')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()

    plt.plot(x, dt_zloss.iloc[:, 0], label='existent')
    plt.plot(x, dt_zloss.iloc[:, 1], label='nonexistent')
    plt.plot(x, dt_zloss.iloc[:, 2], label='all')
    plt.plot(x, best_zloss, label=f'alpha:{alpha:.2f}')
    ymax2 = max(dt_zloss.iloc[:, 2])
    xpos2 = dt_zloss.iloc[:, 2].tolist().index(ymax2)
    xmax2 = x[xpos2]
    plt.text(xmax2, ymax2, f'(x:{xmax2:.3f}, y:{ymax2:.3f}, ex:{dt_zloss.iloc[xpos2, 0]:.3f})', ha='left',
             va='bottom', fontsize=12)
    plt.plot(xmax2, ymax2, 'ro')
    plt.title('Accuracy zero loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()

    plt.text(xmax1, ymax1, f'(x:{xmax1:.3f}, y:{ymax1:.3f}, ex:{dt_highest.iloc[xpos1, 0]:.3f})', ha='left',
             va='bottom', fontsize=12)
    plt.plot(xmax1, ymax1, 'o', label='highest')
    plt.text(xmax2, ymax2, f'(x:{xmax2:.3f}, y:{ymax2:.3f}, ex:{dt_zloss.iloc[xpos2, 0]:.3f})', ha='left',
             va='bottom', fontsize=12)
    plt.plot(xmax2, ymax2, 'o', label='zloss')
    plt.plot(x, dt_highest.iloc[:, 2], label='highest-all')
    plt.plot(x, dt_zloss.iloc[:, 2], label='zloss-all')
    plt.plot(x, dt_highest.iloc[:, 0], label='highest-ex')
    plt.plot(x, dt_zloss.iloc[:, 0], label='zloss-ex')
    plt.title('compare')
    plt.legend()
    plt.grid(True)
    plt.show()

    ymax1 = max(best_highest)
    xpos1 = best_highest.tolist().index(ymax1)
    xmax1 = x[xpos1]
    ymax2 = max(best_zloss)
    xpos2 = best_zloss.tolist().index(ymax2)
    xmax2 = x[xpos2]
    plt.text(xmax1, ymax1, f'(x:{xmax1:.3f}, y:{ymax1:.3f}, ex:{dt_highest.iloc[xpos1, 0]:.3f})', ha='left',
             va='bottom', fontsize=12)
    plt.plot(xmax1, ymax1, 'o', label='highest')
    plt.text(xmax2, ymax2, f'(x:{xmax2:.3f}, y:{ymax2:.3f}, ex:{dt_zloss.iloc[xpos2, 0]:.3f})', ha='left',
             va='bottom', fontsize=12)
    plt.plot(xmax2, ymax2, 'o', label='zloss')
    plt.plot(x, best_highest, label=f'highest alpha:{alpha:.2f}')
    plt.plot(x, best_zloss, label=f'zloss alpha:{alpha:.2f}')
    plt.plot(x, dt_highest.iloc[:, 0], label='highest-ex')
    plt.plot(x, dt_zloss.iloc[:, 0], label='zloss-ex')
    plt.title('compare')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    analysis(0.9)
    # main('../test/DataSet/MyDataSet9-3.csv', '../test/DataSet/9-3.csv')
    # main('../test/DataSet/MyDataSet18-4.csv', '../test/DataSet/18-4.csv')
