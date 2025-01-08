import argparse
from pycocotools.coco import COCO
import json
from detectron2.structures import BoxMode
import numpy as np
from pycocotools._mask import iou
from torchvision.ops import boxes as box_ops
import torch
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Apply class agnostic NMS')
parser.add_argument('--ps', required=True, help='path to the pseudo annotation data')


def save_coco(coco, anns_all, save_name):
    save_dict = {}
    for k in coco.dataset.keys():
        if k != 'annotations' and k!= 'images':
            save_dict[k] = coco.dataset[k]
    # Add annotations
    save_dict['annotations'] = anns_all
    # Add images
    anotaciones_filtradas_imgs_ids = list(set([a['image_id'] for a in anns_all]))
    anotaciones_filtradas_imgs = coco.loadImgs(anotaciones_filtradas_imgs_ids)
    save_dict['images'] = anotaciones_filtradas_imgs

    print(save_name)
    with open(save_name, 'w') as fp:
        json.dump(save_dict, fp, indent=4, sort_keys=True)

    return save_name


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


def nms(coco_dt, th=0.25):
    final_boxes = []
    img_ids = coco_dt.getImgIds()

    # Apply NMS class-agnostic
    for img_id in tqdm(img_ids):
        annotations_id = coco_dt.getAnnIds(imgIds=[img_id])
        annotations = coco_dt.loadAnns(annotations_id)
        if len(annotations) == 0:
            continue
        
        boxes = BoxMode.convert(np.array([a['bbox'] for a in annotations]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS)
        boxes = torch.tensor(boxes)

        scores = torch.tensor(np.array([a['score'] for a in annotations]))
        idxs = torch.tensor(np.array([1] * len(annotations)))

        res = box_ops.batched_nms(boxes.float(), scores.float(), idxs, th).tolist()

        # Add pseudos that pass NMS
        for indice in res:
            final_boxes.append(annotations[indice])

    return final_boxes



def main(args):
    coco_ps = COCO(args.ps)
    print(f"Initial # pseudos: {len(coco_ps.dataset['annotations'])}")
    # NMS clas-agnostic
    dets_to_ignore_anns_score_nms = nms(coco_ps, th=0.25)
    print(f"Final # pseudos: {len(dets_to_ignore_anns_score_nms)}")
    print(f"Number of removed pseudos: {len(coco_ps.dataset['annotations']) - len(dets_to_ignore_anns_score_nms)}")
    save_name = args.ps.replace(".json", "_nms.json")
    save_coco(coco_ps, dets_to_ignore_anns_score_nms, save_name)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)