import numpy as np
from pycocotools._mask import iou
import argparse
from pycocotools.coco import COCO
from tqdm.auto import tqdm
import json
import os


# COCO
UNSEEN_IDS = [ 1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]  # For COCO



# PASCAL VOC categories
PASCAL_VOC_ALL_CATEGORIES = {
    1: ["aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "bird", "bus", "cow", "motorbike", "sofa",
    ],
    2: ["bicycle", "bird", "boat", "bus", "car",
        "cat", "chair", "diningtable", "dog", "motorbike",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "aeroplane", "bottle", "cow", "horse", "sofa",
    ],
    3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
        "car", "chair", "cow", "diningtable", "dog",
        "horse", "person", "pottedplant", "train", "tvmonitor",
        "boat", "cat", "motorbike", "sheep", "sofa",
    ],
}

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}

PASCAL_VOC_BASE_CATEGORIES = {
    1: ["aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
    ],
    2: ["bicycle", "bird", "boat", "bus", "car",
        "cat", "chair", "diningtable", "dog", "motorbike",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
    ],
    3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
        "car", "chair", "cow", "diningtable", "dog",
        "horse", "person", "pottedplant", "train", "tvmonitor",
    ],
}


parser = argparse.ArgumentParser(description='Filter pseudoannotations by removing base known ones')
parser.add_argument('--gt', required=True, help='path to the gt data')
parser.add_argument('--dets', required=True, help='path to the dets data')
parser.add_argument('--th', required=True, help='IoU threshold')
parser.add_argument('--tc', required=True, default="True", choices=["True", "False"], help='Transform continuous cats to real cats ids.')
parser.add_argument('--voc', required=True, default="False", choices=["True", "False"], help='VOC dataset')



def save_coco(coco_dt, filtradas_anns, filename_dt):
    save_dict = {}
    for k in coco_dt.dataset.keys():
        if k != 'annotations' and k!= 'images':
            save_dict[k] = coco_dt.dataset[k]
    
    # Add anotations
    save_dict['annotations'] = filtradas_anns
    # Add images
    anotaciones_filtradas_imgs_ids = list(set([a['image_id'] for a in filtradas_anns]))
    anotaciones_filtradas_imgs = coco_dt.loadImgs(anotaciones_filtradas_imgs_ids)
    save_dict['images'] = anotaciones_filtradas_imgs
    
    save_name = filename_dt.replace('.json', '_filtrado_base2.json')
    print(save_name)
    with open(save_name, 'w') as fp:
        s = json.dumps(save_dict, indent=4, sort_keys=True)
        fp.write(s)



def transformar_categoria(coco_dt, voc, split=None):
    anotaciones = coco_dt.dataset['annotations']
    categorias_continuas = set([x['category_id'] for x in anotaciones])
    
    if not voc:
        # Mapper for COCO
        mapper = {k:j for k,j in zip(categorias_continuas, UNSEEN_IDS)}
    else:
        # Mapper for VOC
        indexes_ = [PASCAL_VOC_ALL_CATEGORIES[split].index(x) for x in PASCAL_VOC_NOVEL_CATEGORIES[split]]
        mapper = {k:j for k,j in zip(categorias_continuas, indexes_)}

    # Updating
    for anot in anotaciones:
        anot['category_id'] = mapper[anot['category_id']]

    


def iou_check(dt_id, coco_dt, coco_gt, thresh=0.9, not_seen_ids=UNSEEN_IDS, voc=False):
    """
    Comprueba si una pseudoetiqueta (que siempre esta etiquetada como clase nueva), tiene un iou superior a th con
    una det de clase base.
    
    Return
    ------
    False: valid pseudo
    True: invalid pseudo
    """

    ann = coco_dt.loadAnns(dt_id)[0]

    d_id = ann['image_id']
    if voc:
        d_id = [d_id]
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=d_id, iscrowd=False))
    # Remove UNSEEN GT (we do not know them)
    anns_gt = [x for x in anns_gt if x['category_id'] not in not_seen_ids]

    if len(anns_gt):
        bbox_dt = np.array([ann['bbox']])
        bbox_gt = np.array([a['bbox'] for a in anns_gt])
        ious = iou(bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()
        
        return ious.max() > thresh
    else:
        return False


def main(filename_dt, filename_gt, th, tc, voc):
    split=None
    if voc:
        split = int(os.path.basename(filename_gt)[-6])


    coco_gt = COCO(filename_gt)
    try:
        coco_dt = COCO(filename_dt)
    except:
        coco_dt = coco_gt.loadRes(filename_dt)

    # Update category id from continuos (1-20 in COCO) to real (UNSEEN IDS):
    if tc:
        transformar_categoria(coco_dt, voc, split)

    dt_ids = coco_dt.getAnnIds()
    print(f"Numero de pseudoetiquetas: {len(dt_ids)}\n")

    dt_ids_filtradas = []
    for dt_id in tqdm(dt_ids, total=len(dt_ids)):

        # Obtain the ids of unknown categories
        if not voc:
            not_seen_ids = UNSEEN_IDS
        else:
            not_seen_ids = [PASCAL_VOC_ALL_CATEGORIES[split].index(x) for x in PASCAL_VOC_NOVEL_CATEGORIES[split]]


        if not iou_check(dt_id, coco_dt, coco_gt, thresh=th, not_seen_ids=not_seen_ids, voc=voc):
            dt_ids_filtradas.append(dt_id)
    
    print(f"Pseudos despues del filtrado (que no eran anot de clases base): {len(dt_ids_filtradas)}")
    print(f"Pseudos eliminadas: {len(dt_ids) - len(dt_ids_filtradas)}")

    filtradas_anns = coco_dt.loadAnns(dt_ids_filtradas)
    save_coco(coco_dt, filtradas_anns, filename_dt)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dets, args.gt, float(args.th), args.tc=='True', args.voc=='True')