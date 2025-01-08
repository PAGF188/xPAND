import numpy as np
from tabulate import tabulate
import contextlib
import io
from pycocotools.coco import COCO
from collections import defaultdict
from tqdm.auto import tqdm
from pycocotools._mask import iou
import argparse
from detectron2.structures import BoxMode, Boxes
from detectron2.structures.boxes import pairwise_ioa


parser = argparse.ArgumentParser(description='Analyze pseudo-labels')

parser.add_argument('--dets', required=True, help='path to the dets annotation data')
parser.add_argument('--gt', required=False, help='path to the gt data', default="datasets/cocosplit/datasplit/trainvalno5k.json")


####################################################################################
####################################################################################
####################################################################################

def print_precision_per_class(filename_dt, filename_gt, iou_thresh=0.5):
    print(f"Precision {iou_thresh} IoU")
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
    # print(filename_dt)
    precisions = defaultdict(list)
    dt_ids = coco_dt.getAnnIds()
    for i, dt_id in tqdm(enumerate(dt_ids), total=len(dt_ids)):
        cid = coco_dt.loadAnns(dt_id)[0]['category_id']
        tp = iou_check(dt_id, coco_dt, coco_gt, thresh=iou_thresh)
        precisions[cid].append(tp)
    table = print_results_table(precisions, coco_gt)
    print(f"No. pseudos: {np.sum([v[1] for v in table])}")
    print(f"Mean precision: {np.mean([v[2] for v in table])}")
    
    return table

def print_results_table(precisions, coco_gt):
    table = []
    cat_ids = sorted(precisions.keys())
    for cid in cat_ids:
        val = np.array(precisions[cid])
        val = val[val != -1]
        name = coco_gt.cats[cid]['name']
        table.append(
            [name, len(val), np.array(val).mean()])
    print(tabulate(
        table, headers=['Category', 'NUM', 'Precision'], tablefmt='orgtbl'))
    return table


def cla_metric(filename_dt, filename_gt, iou_thresh=0.5):
    print(f"Clasification metric a: {iou_thresh} IoU")
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

    precisions = defaultdict(list)
    dt_ids = coco_dt.getAnnIds()
    for i, dt_id in tqdm(enumerate(dt_ids), total=len(dt_ids)):
        cid = coco_dt.loadAnns(dt_id)[0]['category_id']
        tp = iou_check3(dt_id, coco_dt, coco_gt, thresh=iou_thresh)
        precisions[cid].append(tp)
    table = print_results_table(precisions, coco_gt)
    print(f"No. pseudos: {np.sum([v[1] for v in table])}")
    print(f"Cla metric mean: {np.mean([v[2] for v in table])}")



def iou_check(dt_id, coco_dt, coco_gt, thresh=0.5):
    ann = coco_dt.loadAnns(dt_id)[0]
    if 'ignore_qe' in ann:
        if ann['ignore_qe']:
            return -1
    iid, cid = ann['image_id'], ann['category_id']
    if isinstance(iid, str):
        iid = [iid]
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid, catIds=cid, iscrowd=False))
    if len(anns_gt):
        bbox_dt = np.array([ann['bbox']])
        bbox_gt = np.array([a['bbox'] for a in anns_gt])
        ious = iou(bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()
        return 1 if ious.max() > thresh else 0
    else:
        return 0
    

####################################################################################
####################################################################################
####################################################################################
def compute_iou(filename_dt, filename_gt):
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
    # print(filename_dt)
    ious = []
    ioas = []
    dt_ids = coco_dt.getAnnIds()
    for i, dt_id in tqdm(enumerate(dt_ids), total=len(dt_ids)):
        cid = coco_dt.loadAnns(dt_id)[0]['category_id']
        # IoU
        iou = iou_check2(dt_id, coco_dt, coco_gt)
        ious.append(iou)
        #IoA
        ioa = ioa_check(dt_id, coco_dt, coco_gt)
        if ioa != -1:
            ioas.append(ioa)
    
    return ious, ioas

def iou_check2(dt_id, coco_dt, coco_gt, thresh=0.5):
    # TO obtain IoU btw  pseudo and best GT (most overlap)
    ann = coco_dt.loadAnns(dt_id)[0]
    
    iid, cid = ann['image_id'], ann['category_id']
    if isinstance(iid, str):
        iid = [iid]
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid, iscrowd=False))
    if len(anns_gt):
        bbox_dt = np.array([ann['bbox']])
        bbox_gt = np.array([a['bbox'] for a in anns_gt])

        ious = iou(bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()
        return ious.max()
    else:
        return 0

def iou_check3(dt_id, coco_dt, coco_gt, thresh=0.5):
    # From all pseudos with IoU>th with GT (not necessarily of the same class) 
    # check if class is correct
    ann = coco_dt.loadAnns(dt_id)[0]
    
    iid, cid = ann['image_id'], ann['category_id']
    if isinstance(iid, str):
        iid = [iid]
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid, iscrowd=False))
    if len(anns_gt):
        bbox_dt = np.array([ann['bbox']])
        bbox_gt = np.array([a['bbox'] for a in anns_gt])
        ious = iou(bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()                
        iou_max_index = ious.argmax()
        iou_max_v = ious.max()
        if iou_max_v > thresh:
            if cid == anns_gt[iou_max_index]['category_id']:
                return 1
            else:
                return 0
        else:
            return -1
    else:
        return -1



def ioa_check(dt_id, coco_dt, coco_gt, thresh=0.5):
    ann = coco_dt.loadAnns(dt_id)[0]
    
    iid, cid = ann['image_id'], ann['category_id']
    if isinstance(iid, str):
        iid = [iid]
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid, catIds=cid, iscrowd=False))
    if len(anns_gt):
        bbox_dt = BoxMode.convert(
            np.array([ann['bbox']]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS)
        bbox_gt = BoxMode.convert(
            np.array([a['bbox'] for a in anns_gt]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS)
        ioas = pairwise_ioa(Boxes(bbox_gt), Boxes(bbox_dt)).numpy()
        return np.max(ioas)
    else:
        return -1


def compute_scores(filename_dt):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = COCO(filename_dt)
        scores = [x['score'] for x in coco_dt.dataset['annotations']]
        return scores

def compute_hist(array, step=0.1, end=1):
    hist = []
    no_bins = int(end / step)
    for i in range(no_bins):
        hist.append(np.sum((array >= step*i) & (array <= step*i+step)))
    return hist, (np.array(hist) / np.sum(np.array(hist))).tolist() 


def compute_scores_per_class(filename_dt):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = COCO(filename_dt) 

    scores_per_class = {x:[] for x in set([x['category_id'] for x in coco_dt.dataset['annotations']])}
    for anot in coco_dt.dataset['annotations']:
        scores_per_class[anot['category_id']].append(anot['score'])
    
    return scores_per_class

def compute_iou_per_class(filename_dt, filename_gt):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = COCO(filename_dt) 
        coco_gt = COCO(filename_gt)

        
    ious_per_class = {x:[] for x in set([x['category_id'] for x in coco_dt.dataset['annotations']])}

    dt_ids = coco_dt.getAnnIds()
    for i, dt_id in tqdm(enumerate(dt_ids), total=len(dt_ids)):
        cid = coco_dt.loadAnns(dt_id)[0]['category_id']
        # IoU
        iou = iou_check2(dt_id, coco_dt, coco_gt)
        ious_per_class[cid].append(iou)

    return ious_per_class


def main(args):
    
    # 0) Clasification metic. ignoring IoU with gt (>th)
    #cla_metric(args.dets, args.gt, 0.25)

    # #########################################################################
    # # 1) Precision
    # #########################################################################
    print_precision_per_class(args.dets, args.gt, 0.5)
    print_precision_per_class(args.dets, args.gt, 0.75)
    print_precision_per_class(args.dets, args.gt, 0.25)

    
    # #########################################################################
    # # 2) IoU, IoA
    # #########################################################################
    ious, ioas = compute_iou(args.dets, args.gt)

    ious = np.array(ious)
    print(f"IoU medio: {np.mean(ious)}")

    ioas = np.array(ioas)
    print(f"IoA medio: {np.mean(ioas)}")

    hist_absolute_iou, hist_norm_iou = compute_hist(ious, step=0.1, end=1)
    print(f"Histograma IoU abs: {hist_absolute_iou}")
    print(f"Histograma IoU nor: {hist_norm_iou}\n")
    


    # #########################################################################
    # # 3) HIST SCORES
    # #########################################################################
    scores = compute_scores(args.dets)
    scores = np.array(scores)
    hist_absolute_score, hist_norm_score = compute_hist(scores, step=0.1, end=1)
    print(f"Histograma Score abs: {hist_absolute_score}")
    print(f"Histograma Score nor: {hist_norm_score}")

    
    #########################################################################
    # 4) PER CLASS
    #########################################################################
    # scores_por_clase = compute_scores_per_class(args.dets)
    # for k_ in scores_por_clase.keys():
    #     print(f"Class: {k_}")
    #     h, h_norm = compute_hist(np.array(scores_por_clase[k_]), step=0.1, end=1)
    #     print(f"Histograma Score abs: {h}")
    #     print(f"Histograma Score nor: {h_norm}")

    # iou_por_clase = compute_iou_per_class(args.dets, args.gt)
    # for k_ in iou_por_clase.keys():
    #     print(f"Class: {k_}")
    #     h, h_norm = compute_hist(np.array(iou_por_clase[k_]), step=0.1, end=1)
    #     print(f"Histograma Score abs: {h}")
    #     print(f"Histograma Score nor: {h_norm}")
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)