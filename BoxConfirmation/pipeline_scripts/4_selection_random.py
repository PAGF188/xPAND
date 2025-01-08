import argparse
import json
from pycocotools.coco import COCO
import random
import numpy as np


parser = argparse.ArgumentParser(description='Compute IoU')
parser.add_argument('--dets', required=True, help='path to the dets data')
parser.add_argument('--n', required=False, help='# pseudos per cat')
parser.add_argument('--factor', required=False, help='factor desbalanceo')


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

def contar(coco):
    CATS = set([x['category_id'] for x in coco.dataset['annotations']])
    n_anots_per_cat = {x:0 for x in CATS}
    for anot in coco.dataset['annotations']:
        n_anots_per_cat[anot['category_id']] += 1

    return n_anots_per_cat
    



def main(args):
    coco = COCO(args.dets)
    CATS = set([x['category_id'] for x in coco.dataset['annotations']])
    final_anotations = []
    # MInimum number
    n_anots_per_cat = contar(coco)
    n_anots_per_cat_abs = np.array(list(n_anots_per_cat.values()))
    min_ = n_anots_per_cat_abs.min()
    
    min_value = np.maximum(min_ * int(args.factor), 500)

    for cat in CATS:
        pseudo_ids = coco.getAnnIds(catIds=cat) 
        pseudo_ids_anns = coco.loadAnns(pseudo_ids)

        elements = random.sample(pseudo_ids_anns, np.minimum(int(min_value), len(pseudo_ids_anns)))
        final_anotations += elements
    
    save_name = args.dets.replace(".json", f"_random_selection.json")
    save_coco(coco, final_anotations, save_name)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)