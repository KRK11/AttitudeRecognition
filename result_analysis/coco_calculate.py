'''
**************************************************
@File   ：AttitudeRecognition -> coco_calculate
@IDE    ：PyCharm
@Author ：TheOnlyMan
@Date   ：2023/5/30 14:21
**************************************************
'''

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io2
import pylab,json
import matplotlib.pyplot as plt


cocoGt = COCO('/kaggle/input/coco-2017-dataset/coco2017/annotations/person_keypoints_val2017.json')
cocoDt = cocoGt.loadRes('/kaggle/working/test/2023-05-30_16-23-02/result.json')
cocoEval = COCOeval(cocoGt, cocoDt, "keypoints")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
print(cocoEval.eval['precision'].shape)
pr_array1 = cocoEval.eval['precision'][5, :, 0, 0, 0]
pr_array2 = cocoEval.eval['precision'][5, :, 0, 1, 0]
pr_array3 = cocoEval.eval['precision'][5, :, 0, 2, 0]
x = np.arange(0.0, 1.01, 0.01)
recall = cocoEval.params.recThrs
x = recall
print(x)
plt.xlabel('recall')
plt.ylabel('precision')
plt.xlim(0, 1.0)
plt.ylim(0, 1.01)
plt.grid(True)

plt.plot(x, pr_array1, 'b-', label='all')
plt.plot(x, pr_array2, 'c-', label='medium')
plt.plot(x, pr_array3, 'y-', label='large')
plt.title("OKS 0.75")

plt.legend(loc="lower left")
plt.show()

# (10, 101, 1, 3, 1)
#  OKS,Recall,..,All/Medium/Large,..
# you can achieve others.