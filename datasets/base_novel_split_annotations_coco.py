import argparse
from pycocotools.coco import COCO
import numpy as np
import json

# FOR COCO
UNSEEN_IDS = [ 1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
SEEN_IDS = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

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


def main(gt):
    coco_gt = COCO(gt)

    # Base class annotations
    ids = coco_gt.getAnnIds(catIds=SEEN_IDS)
    print(f"Number of base detections: {len(ids)}")
    ids_anns = coco_gt.loadAnns(ids)
    save_name = gt.replace(".json", "_base.json")
    save_coco(coco_gt, ids_anns, save_name)

    # Novel class annotations
    ids = coco_gt.getAnnIds(catIds=UNSEEN_IDS)
    print(f"Number of novel detections: {len(ids)}")
    ids_anns = coco_gt.loadAnns(ids)
    save_name = gt.replace(".json", "_novel.json")
    save_coco(coco_gt, ids_anns, save_name)


def parse_args():
    parser = argparse.ArgumentParser(description='Split base/novel annotations')
    parser.add_argument('--gt', required=True, help='Path to annotations json')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.gt)