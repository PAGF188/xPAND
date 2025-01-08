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


parser = argparse.ArgumentParser(description='')
parser.add_argument('--gt', required=False, help='path to the gt data')
parser.add_argument('--dets', required=True, help='path to the dets data')
parser.add_argument('--finetuning', default=False, action='store_true', help='If finetuning, the selection process is different. Iou is already computed. Selection per class')

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



def print_hist_dist(hist):
    for k in hist.keys():
        print(f"{k}: {len(hist[k])}")


def iou_check2(dt_id, coco_dt, coco_gt):
    ann = coco_dt.loadAnns(dt_id)[0]
    
    iid = ann['image_id']
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[iid], iscrowd=False))
    if len(anns_gt):
        bbox_dt = np.array([ann['bbox']])
        bbox_gt = np.array([a['bbox'] for a in anns_gt])

        ious = iou(bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()
        return ious.max()
    else:
        return 0


def compute_iou(coco_dt, coco_gt):
    
    ious = {}
    dt_ids = coco_dt.getAnnIds()
    for dt_id in tqdm(dt_ids, total=len(dt_ids)):
        # IoU
        iou = iou_check2(dt_id, coco_dt, coco_gt)
        ious[dt_id] = iou
    
    return ious


def compute_hist(ious, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    
    hist = {x:[] for x in range(len(bins))}
    for id in tqdm(ious.keys()):
        iou_ = ious[id]
        tt = bisect.bisect(bins, iou_) - 1
        hist[tt].append(id)
    
    return hist 


def selection_finetuning(coco_dt, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], size=500, test_percentaje=0.1):
    clases = set([x['category_id'] for x in coco_dt.dataset['annotations']])
    anots_finales_train = []
    anots_finales_test = []

    for c in clases:
        hist = {x:[] for x in range(len(bins))}
        anots_c_ids = coco_dt.getAnnIds(catIds=[c], iscrowd=None)
        anots_c = coco_dt.loadAnns(anots_c_ids)
        for anot in anots_c:
            iou_ = anot['iou']
            tt = bisect.bisect(bins, iou_) - 1
            hist[tt].append(anot['id'])
        
        # Selection
        #print_hist_dist(hist)
        total_ids_train = []
        total_ids_test = []
        for b in hist.keys():
            total_ids_b = random.sample(hist[b], np.minimum(size, len(hist[b])))
            test_ids_b = random.sample(total_ids_b, int(test_percentaje * len(total_ids_b)))
            train_ids_b = [x for x in total_ids_b if x not in test_ids_b]

            total_ids_test += test_ids_b
            total_ids_train += train_ids_b
        

        coco_dt_only_selected_train = coco_dt.loadAnns(total_ids_train)
        anots_finales_train += coco_dt_only_selected_train

        coco_dt_only_selected_test = coco_dt.loadAnns(total_ids_test)
        anots_finales_test += coco_dt_only_selected_test

    return anots_finales_train, anots_finales_test


def seleccion(hist, coco_dt, size, ious, test_s=1000):
    total_ids_train = []
    total_ids_test = []
    for b in hist.keys():
        total_ids = random.sample(hist[b], np.minimum(size, len(hist[b])))
        # Split in train test
        total_ids_test += random.sample(total_ids, test_s)
        total_ids_train += [x for x in total_ids if x not in total_ids_test]
    
    coco_dt_only_selected_train = coco_dt.loadAnns(total_ids_train)
    coco_dt_only_selected_test = coco_dt.loadAnns(total_ids_test)


    # Add IoU value
    for x in coco_dt_only_selected_train:
        x['iou'] = ious[x['id']]
    for x in coco_dt_only_selected_test:
        x['iou'] = ious[x['id']]

    return coco_dt_only_selected_train, coco_dt_only_selected_test



def main(args):
    random.seed(88)

    if not args.finetuning:
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

        ious = compute_iou(coco_dt, coco_gt)
        hist = compute_hist(ious)
        print_hist_dist(hist)
        # Selection values
        max_value = 40000
        min_values = np.minimum(max_value, min([len(hist[x]) for x in hist.keys()]))
        test_elements = int(0.1 * min_values)
        coco_dt_only_selected_train, coco_dt_only_selected_test = seleccion(hist, coco_dt, min_values, ious, test_elements)

        save_coco(coco_dt, coco_dt_only_selected_train, filename_dt.replace(".json", f"_box_selected_train.json"))
        save_coco(coco_dt, coco_dt_only_selected_test, filename_dt.replace(".json", f"_box_selected_test.json"))

    else:
        filename_dt = args.dets
        coco_dt = COCO(filename_dt)
        print(filename_dt)
        coco_dt_only_selected_train, coco_dt_only_selected_test = selection_finetuning(coco_dt)
        
        save_coco(coco_dt, coco_dt_only_selected_train, filename_dt.replace(".json", f"_box_selected_train.json"))
        save_coco(coco_dt, coco_dt_only_selected_test, filename_dt.replace(".json", f"_box_selected_test.json"))




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)