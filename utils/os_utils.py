'''
**************************************************
@File   ：AttitudeRecognition -> os_utils
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/4/27 12:04
**************************************************
'''

import os
import sys
import errno
from datetime import datetime


def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def newdir(dir):
    now = datetime.now()
    path = os.path.join(dir, now.strftime('%Y-%m-%d_%H-%M-%S'))
    mkdir_p(path)
    return path


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
