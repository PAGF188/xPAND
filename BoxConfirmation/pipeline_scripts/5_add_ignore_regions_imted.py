import argparse
from pycocotools.coco import COCO
import json
from detectron2.structures import BoxMode
import numpy as np
from pycocotools._mask import iou
from torchvision.ops import boxes as box_ops
import torch
parser = argparse.ArgumentParser(description='Combine pseudo-annotations with a ignore dataset')
parser.add_argument('--ps', required=True, help='path to the pseudo annotation data')
parser.add_argument('--ig', required=True, help='path to the ignore data')
parser.add_argument('--score', required=True, help='score para considerar una ignore region')


def intersect(bbox_dt, box_):
    xA = max(bbox_dt[0][0], box_[0])
    yA = max(bbox_dt[0][1], box_[1])
    xB = min(bbox_dt[0][2], box_[2])
    yB = min(bbox_dt[0][3], box_[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea

def area(box):
    area = np.abs(box[0] - box[2]) * np.abs(box[1] - box[3])
    return area

def i_check(dt_ann, coco_pseudos, thresh=0.25):
    """
    Return
    ------
    True: Deteccion que no tiene IoU>thresh con pseudo -> es valida para ignore region
    False: Lo contrario
    """

    d_id = dt_ann['image_id']
    if isinstance(d_id, str):
        d_id = [d_id]
    anns_pseudos = coco_pseudos.loadAnns(coco_pseudos.getAnnIds(imgIds=d_id, iscrowd=False))
    if len(anns_pseudos):
        bbox_dt = BoxMode.convert(
            np.array([dt_ann['bbox']]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS)
        bbox_gt = BoxMode.convert(
            np.array([a['bbox'] for a in anns_pseudos]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS)
        
        for box_ in bbox_gt:
            intersecion = intersect(bbox_dt, box_)
            if intersecion > 0:
                pseudo_over_inter = intersecion / area(box_)
                if pseudo_over_inter >= thresh:
                    return False

        return True
    else:
        raise AssertionError (f"Deberia haber pseudo en image: {d_id}")


def aplicar_nms(dets_to_ignore_anns_score, th=0.25):
    ignore_regions_finales = []

    imgs_keys = set([x['image_id'] for x in dets_to_ignore_anns_score])
    dict_imgs = {x: [] for x in imgs_keys}
    for det in dets_to_ignore_anns_score:
        dict_imgs[det['image_id']].append(det)

    # NMS cat agnostic 
    for img_id in dict_imgs.keys():
        annotations = dict_imgs[img_id]
        boxes = BoxMode.convert(np.array([a['bbox'] for a in annotations]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS)
        boxes = torch.tensor(boxes)

        scores = torch.tensor(np.array([a['score'] for a in annotations]))
        idxs = torch.tensor(np.array([1] * len(annotations)))

        res = box_ops.batched_nms(boxes.float(), scores.float(), idxs, th).tolist()

        # Add the ones that has not pass NMS
        for indice in res:
            ignore_regions_finales.append(annotations[indice])

    return ignore_regions_finales



def save_coco(coco, anns_all, filename_dt, score):

    save_dict = {}
    for k in coco.dataset.keys():
        if k != 'annotations' and k!= 'images':
            save_dict[k] = coco.dataset[k]
    save_dict['annotations'] = anns_all
    anotaciones_filtradas_imgs_ids = list(set([a['image_id'] for a in anns_all]))
    anotaciones_filtradas_imgs = coco.loadImgs(anotaciones_filtradas_imgs_ids)
    save_dict['images'] = anotaciones_filtradas_imgs

    save_name = filename_dt.replace('.json', f'_ignore_regions{score}_interseccion_nms.json')
    print(save_name)
    with open(save_name, 'w') as fp:
        s = json.dumps(save_dict, indent=4, sort_keys=True)
        fp.write(s)



def main(args):
    coco_ps = COCO(args.ps)
    coco_ig = COCO(args.ig)


    # We are going to complete the coco_ps images with coco_ig detections that 
    # are not in coco_ps and that exceed a certain score threshold.
    
    # We obtain id imgs to complete
    imgs_ids = coco_ps.getImgIds()

    # We obtain the id of the detections present in these images to ignore.
    dets_ig_ids = coco_ig.getAnnIds(imgIds=imgs_ids, iscrowd=False)
    # We obtain the id of the detections not to ignore
    dets_ps_ids = coco_ps.getAnnIds(imgIds=imgs_ids, iscrowd=False)

    # Possible detections to ignore
    dets_to_ignore_ids = set(dets_ig_ids) - set(dets_ps_ids)
    

    # We load annotations of these detections to ignore
    dets_to_ignore_anns = coco_ig.loadAnns(dets_to_ignore_ids)
    # We filter by score; and if there is no intersection > 0.25 with detection
    dets_to_ignore_anns_score = [x for x in dets_to_ignore_anns if x['score'] > float(args.score) and i_check(x, coco_ps, thresh=0.25)]
    
    # We apply NMS per image. Without considering category
    dets_to_ignore_anns_score_nms = aplicar_nms(dets_to_ignore_anns_score, th=0.25)

    # Finally we add these detections to the annotation file
    anot_aux = []
    for anot in dets_to_ignore_anns_score_nms:
        anot['ignore'] = 1   #  IMTED
        anot_aux.append(anot)
    
    anot_aux2 = []
    for anot in coco_ps.loadAnns(dets_ps_ids):
        anot['ignore'] = 0  # IMTED
        anot_aux2.append(anot)

    anot_finales = anot_aux2 + anot_aux
    
    save_coco(coco_ps, anot_finales, args.ps, float(args.score))




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)