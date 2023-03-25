import json

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# accumulate predictions from all images
# 载入coco2017验证集标注文件

val_json_path_coco_format=r'D:/dataset/data/new_dataset/annotations/test.json'
# val_json_path_coco_format=r'D:/dataset/data/new_dataset/annotations/test.json'
coco_true = COCO(annotation_file=val_json_path_coco_format)
# 载入网络在coco2017验证集上预测的结果
result_json_path=r'D:\dataset\实验记录\46\test\yolov7x2\best_predictions.json'
coco_pre = coco_true.loadRes(result_json_path)

coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
# Get the index of the IoU threshold of 0.5
iou_idx = np.where(coco_evaluator.params.iouThrs == 0.5)[0][0]
precision = coco_evaluator.eval['precision'][iou_idx, :, :, 2].mean()
recall = coco_evaluator.eval['recall'][iou_idx, :, :, 2].mean()
print(precision)
print(recall)
print("=====================")

def mean_recall(coco_eval):
    recall_values = []
    precision_values=[]
    for iou_idx in range(len(coco_eval.params.iouThrs)):
        recall = coco_eval.eval['recall'][iou_idx, :, :, 2]
        precision = coco_eval.eval['precision'][iou_idx, :, :, 2]
        recall_values.append(recall)
        precision_values.append(precision)

    return np.mean(recall_values),np.mean(precision_values)

mean_r,mean_p=mean_recall(coco_evaluator)
print("mean_recall:",mean_r)
print("mean_recall:",mean_p)


coco_evaluator.summarize()
