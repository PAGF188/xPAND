import argparse
from pycocotools.coco import COCO
from tqdm.auto import tqdm
import numpy as np
from pycocotools._mask import iou
import json
from random import sample
import random
import io
import contextlib
import bisect


parser = argparse.ArgumentParser(description='Compute IoU')
parser.add_argument('--gt', required=True, help='path to the gt data')
parser.add_argument('--dets', required=True, help='path to the dets data')


def save_coco(coco, anns_all, save_name):
    save_dict = {}
    for k in coco.dataset.keys():
        if k != 'annotations' and k!= 'images':
            save_dict[k] = coco.dataset[k]
    save_dict['annotations'] = anns_all
    anotaciones_filtradas_imgs_ids = list(set([a['image_id'] for a in anns_all]))
    anotaciones_filtradas_imgs = coco.loadImgs(anotaciones_filtradas_imgs_ids)
    save_dict['images'] = anotaciones_filtradas_imgs

    print(save_name)
    with open(save_name, 'w') as fp:
        json.dump(save_dict, fp, indent=4, sort_keys=True)

    return save_name


def iou_appender(dt_id, coco_dt, coco_gt):
    ann = coco_dt.loadAnns(dt_id)[0]
    
    iid = ann['image_id']
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid, iscrowd=False))
    if len(anns_gt):
        bbox_dt = np.array([ann['bbox']])
        bbox_gt = np.array([a['bbox'] for a in anns_gt])

        ious = iou(bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()
        ann['iou'] = ious.max()
    else:
        ann['iou'] = 0

    return ann


def compute_iou(coco_dt, coco_gt):
    
    annotations = []
    dt_ids = coco_dt.getAnnIds()
    for dt_id in tqdm(dt_ids, total=len(dt_ids)):
        # IoU
        anot = iou_appender(dt_id, coco_dt, coco_gt)
        annotations.append(anot)
    return annotations


def main(args):
    random.seed(88)
    filename_dt = args.dets
    filename_gt = args.gt

    with contextlib.redirect_stdout(io.StringIO()):
        if isinstance(filename_gt, str):
            coco_gt = COCO(filename_gt)
        else:
            coco_gt = filename_gt
        if isinstance(filename_dt, str):
            print(filename_dt)
            try:
                coco_dt = COCO(filename_dt)
            except:
                coco_dt = coco_gt.loadRes(filename_dt)
        else:
            coco_dt = filename_dt
    print(filename_dt)

    annotations = compute_iou(coco_dt, coco_gt)
    save_coco(coco_dt, annotations, filename_dt.replace(".json", f"_dummy_iou.json"))




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)