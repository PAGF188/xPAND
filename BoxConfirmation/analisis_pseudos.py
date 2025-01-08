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

# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

parser = argparse.ArgumentParser(
    description='Combine ubbr with pseudo-annotations')

parser.add_argument('--dets', required=True, help='path to the dets annotation data')
parser.add_argument('--gt', required=False, help='path to the gt data', default="datasets/cocosplit/datasplit/trainvalno5k.json")


####################################################################################
####################################################################################
####################################################################################
# Para precision pseudos

def print_precision_per_class(filename_dt, filename_gt, iou_thresh=0.5):
    print(f"Precision a: {iou_thresh} IoU")
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
    buenas = 0
    for i, dt_id in tqdm(enumerate(dt_ids), total=len(dt_ids)):
        cid = coco_dt.loadAnns(dt_id)[0]['category_id']
        tp = iou_check(dt_id, coco_dt, coco_gt, thresh=iou_thresh)
        if tp == 1:
            buenas += 1
        precisions[cid].append(tp)
    table = print_results_table(precisions, coco_gt)
    print(f"No. pseudos: {np.sum([v[1] for v in table])},  buenas: {buenas}")
    print(f"Precision media: {np.mean([v[2] for v in table])}")
    
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
    print(f"Cla metric media: {np.mean([v[2] for v in table])}")



def iou_check(dt_id, coco_dt, coco_gt, thresh=0.5):
    # Para obtener si pseudo esta bien clasificada e IoU>thresh
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
# Para computar IoU and intersection over area
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
    # Para obtener el IoU entre la pseudo y la mejor GT (la de mas solape)
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
    # De todas las pseudos con IoU>th con un GT (no necesariamente de la misma clase) 
    # mirar si la cat es la correcta
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
        ioas = pairwise_ioa(Boxes(bbox_gt), Boxes(bbox_dt)).numpy()  # Interseccion sobre area del segundo argumento
        return np.max(ioas)
    else:
        return -1


def compute_scores(dets):
    scores = [x['score'] for x in dets]
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


def obtain_dets(filename_dt, filename_gt, iou_thresh=0.5):
    print(f"Obtaining pseudos at: {iou_thresh} IoU")
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

    final_dets_ids = []
    dt_ids = coco_dt.getAnnIds()
    for dt_id in tqdm(dt_ids, total=len(dt_ids)):
        tp = iou_check(dt_id, coco_dt, coco_gt, thresh=iou_thresh)
        if tp==1:
            final_dets_ids.append(dt_id)

    final_dets = coco_dt.loadAnns(final_dets_ids)
    return final_dets


def main(args):
    
    # 0) Clasification metic. Clasification acc. ignoring IoU with gt (>th)
    #cla_metric(args.dets, args.gt, 0.25)

    # #########################################################################
    # # 1) Precisiones
    # #########################################################################
    print_precision_per_class(args.dets, args.gt, 0.5)
    print_precision_per_class(args.dets, args.gt, 0.75)
    print_precision_per_class(args.dets, args.gt, 0.25)

    
    # # #########################################################################
    # # # 2) IoU, IoA
    # # #########################################################################
    ious, ioas = compute_iou(args.dets, args.gt)
    
    # sns.set(rc={'figure.figsize':(5,1)})
    # sns.set_theme()
    # sns.set_style("ticks")
    # asd = sns.boxplot(data=ious, orient='h', palette=['white'], showfliers = False, linewidth=2, width=0.33, linecolor='black')
    # asd.set_xlim(-0.05, 1.05)
    # asd.figure.savefig("./boxplot0_scale.png")


    ious = np.array(ious)
    print(f"IoU medio: {np.mean(ious)}")

    ioas = np.array(ioas)
    print(f"IoA medio: {np.mean(ioas)}")

    hist_absolute_iou, hist_norm_iou = compute_hist(ious, step=0.1, end=1)
    print(f"Histograma IoU abs: {hist_absolute_iou}")
    print(f"Histograma IoU nor: {hist_norm_iou}\n")
    
    # Plot boxplot of IoUs



    # # #########################################################################
    # # # 3) HIST SCORES
    # # #########################################################################
    # import json
    # asd = json.load(open(args.dets))
    # if not isinstance(asd, list):
    #     dets = asd['annotations']
    # else:
    #     dets = asd
    
    # scores_a = compute_scores(dets)
    # scores_a = np.array(scores_a)
    # hist_absolute_score_a, hist_norm_score_a = compute_hist(scores_a, step=0.05, end=1)
    # print(f"Histograma Score abs: {hist_absolute_score_a}")
    # print(f"Histograma Score nor: {hist_norm_score_a}")

    # #Obtener scores PR>0.5
    # dets = obtain_dets(args.dets, args.gt, 0.5)
    # scores_d = compute_scores(dets)
    # scores_d = np.array(scores_d)
    # hist_absolute_score_d, hist_norm_score_d = compute_hist(scores_d, step=0.05, end=1)
    # print(f"Histograma Score abs: {hist_absolute_score_d}")
    # print(f"Histograma Score nor: {hist_norm_score_d}")

    # asd = sns.barplot(data=hist_absolute_score_a[10:], color="#ff9f9b", width=0.5)
    # asd = sns.barplot(data=hist_absolute_score_d[10:], color="#a1c9f4", width=0.5)
    # asd.figure.savefig("./hist_imted_borrar_d.png")
    
    #########################################################################
    # 4) RESULTADOS POR CLASE
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