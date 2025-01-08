"""
Script to select pseudos with an specific IoU
"""

import argparse
import json
from pycocotools.coco import COCO
from tqdm.auto import tqdm
import contextlib
import io
import numpy as np
from pycocotools._mask import iou

UNSEEN_IDS = [ 1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
parser = argparse.ArgumentParser(description='IoU pseudo selection')
parser.add_argument('--dets', required=True, help='path to the pseudo data')
parser.add_argument('--gt', required=False, help='path to the gt data', default="datasets/cocosplit/datasplit/trainvalno5k.json")
parser.add_argument('--iou', required=True, help='threshold iou')


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



def iou_check2(dt_id, coco_dt, coco_gt, thresh=0.5):
    ann = coco_dt.loadAnns(dt_id)[0]
    iid, cid = ann['image_id'], ann['category_id']
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid, iscrowd=False))

    if len(anns_gt):
        bbox_dt = np.array([ann['bbox']])
        bbox_gt = np.array([a['bbox'] for a in anns_gt])

        ious = iou(bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()
        return (ious.max() >= thresh and ious.max() <= thresh + 0.25)
    else:
        return False


def main(args):
    filename_gt = args.gt
    filename_dt = args.dets
    iou_thresh = float(args.iou)

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
    
    dt_ids = coco_dt.getAnnIds()
    anot_finales = []
    for dt_id in tqdm(dt_ids):
        anot = coco_dt.loadAnns(dt_id)[0]
        if iou_check2(dt_id, coco_dt, coco_gt, thresh=iou_thresh):
            anot_finales.append(anot)

    save_name = filename_dt.replace(".json", f"_iou_{iou_thresh}.json")
    save_coco(coco_dt, anot_finales, save_name)




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
        
