# Based on https://github.com/prannaykaul/lvc


import argparse
from pycocotools.coco import COCO
import numpy as np
import json
from math import ceil

AREA_RNG = [0 ** 2, 1e5 ** 2]


parser = argparse.ArgumentParser(description='Score filtering')
parser.add_argument('--dets', required=True, help='path to the dets data')
parser.add_argument('--K-min', type=float, help='min score for select detections', required=True)
parser.add_argument('--K-max', type=float, help='max score for select detections', required=True)


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



def filtrado(coco_dt, args):
    
    cats_ids = set([x['category_id'] for x in coco_dt.dataset['annotations']])

    # Delete this from coco_dt
    valid_pseudo_imgs_ids = coco_dt.getImgIds()

    all_anns = []
    for cid in cats_ids:
        ann_ids = coco_dt.getAnnIds(catIds=cid, imgIds=valid_pseudo_imgs_ids, areaRng=AREA_RNG, iscrowd=False)
        anns = coco_dt.loadAnns(ann_ids)
        anns = sorted(anns, key=lambda x: x['score'], reverse=True)

        K_min = float(args.K_min)
        K_max = float(args.K_max)
        scores = np.array([x['score'] for x in anns])

        ind_min = np.searchsorted(-scores, -K_min)
        ind_max = np.searchsorted(-scores, -K_max)
            
        keep_anns_aux = anns[ind_max:ind_min]
        scores = np.array([x['score'] for x in keep_anns_aux])

        # 1) -2000 # old code
        #indices_mantener = np.argsort(scores)[np.maximum(-2000, -scores.shape[0]):]

        # 2) Para todas
        indices_mantener = np.argsort(scores)

        # 3) Mantener quantil # old code
        #quantili_size = scores.shape[0] - ceil(0.75 * scores.shape[0])
        #indices_mantener = np.argsort(scores)[np.maximum(-quantili_size, -scores.shape[0]):]


        keep_anns = []
        for index_keep in indices_mantener:
            ann_aux = keep_anns_aux[int(index_keep)]
            ann_aux['ignore_qe'] = 0
            ann_aux['iscrowd'] = 0
            keep_anns.append(ann_aux)
        all_anns.extend(keep_anns)

    return all_anns

def main(args):

    coco_dt = COCO(args.dets)   # Pseudos

    
    anotaciones_filtradas = filtrado(coco_dt, args)
    save_name = args.dets.replace(".json", f"_filt_score{args.K_min}.json") 
    save_coco(coco_dt, anotaciones_filtradas, save_name)
   


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
