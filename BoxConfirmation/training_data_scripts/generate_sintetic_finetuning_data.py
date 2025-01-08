import json
from tqdm import tqdm
import argparse
from pycocotools.coco import COCO
import numpy as np
from pycocotools._mask import iou
import os

GLOBAL_ID = 0
RNG = np.random.default_rng()


parser = argparse.ArgumentParser(description='Add predicted IoU score to coco annotations')
parser.add_argument('--gt', required=True, help='path to the gt data to increment with sint. boxes')
parser.add_argument('--output', required=True, help='output file')

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


def generate_sintetic_boxes(ann, image_size, samples=10000):
    """
    Image size in in (h,w) format
    """

    result_annotations = []
    box = ann['bbox']
    x1, y1, w, h = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
    
    ## Random normal distribution for each
    x1_displacements = RNG.normal(0, 0.2*w, size=samples)
    x2_displacements = RNG.normal(0, 0.2*w, size=samples)
    y1_displacements = RNG.normal(0, 0.2*h, size=samples)
    y2_displacements = RNG.normal(0, 0.2*h, size=samples)


    ## New values.
    x1_news = (x1_displacements + x1).clip(0, image_size[1])
    x2_news = (x2_displacements + x2).clip(0, image_size[1])
    y1_news = (y1_displacements + y1).clip(0, image_size[0])
    y2_news = (y2_displacements + y2).clip(0, image_size[0])

    # Reconvert to XYWH format
    for x1_, y1_, x2_, y2_ in zip(x1_news, y1_news, x2_news, y2_news):
        global GLOBAL_ID
        w = np.abs(x2_ - x1_)
        h = np.abs(y2_ - y1_)
        bbox_ = [x1_, y1_, w, h]

        iou_ = iou((np.array(bbox_).astype(int))[np.newaxis, ...], (np.array(box).astype(int))[np.newaxis, ...], [0]).squeeze()    
        aux_anot = {'area':w*h, 'iscrowd':ann['iscrowd'], 'image_id':ann['image_id'], 'category_id':ann['category_id'], 'id':GLOBAL_ID, 'bbox': bbox_, "iou": float(iou_)}
        result_annotations.append(aux_anot)
        GLOBAL_ID += 1

    return result_annotations


def main(args):

    anots_finales = []
    
    coco = COCO(args.gt)
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        image_info = coco.loadImgs([ann['image_id']])[0]
        result_annotations = generate_sintetic_boxes(ann, (image_info['height'], image_info['width'])) 
        anots_finales += result_annotations
    
    save_name = os.path.join(args.output, os.path.basename(args.gt).replace(".json", f"_sint_boxes.json"))
    print(save_name)
    save_coco(coco, anots_finales, save_name)



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)