import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_true', type=str, default='', help='coco ture json')
    parser.add_argument('result', type=str, default='', help='result json')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def evaluate(opt):
    val_json_path_coco_format=opt.coco_true
    result_json_path=opt.result
    # accumulate predictions from all images
    # 载入coco2017验证集标注文件
    coco_true = COCO(annotation_file=val_json_path_coco_format)
    # 载入网络在coco2017验证集上预测的结果
    coco_pre = coco_true.loadRes(result_json_path)
    coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


if __name__ == '__main__':
    opt = parse_opt()
    evaluate(opt)
