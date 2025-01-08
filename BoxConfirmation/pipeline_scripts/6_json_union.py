# ONLY FOR VOC!!!

import argparse
import json
from pycocotools.coco import COCO
import random
import numpy as np
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ps', required=True, help='path to pseudos')
parser.add_argument('--gt', required=True, help='path to support shot')

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



def main(args):
    pseudos = json.load(open(args.ps))
    gt = json.load(open(args.gt))

    # The pseudos categories are in format 0-20 (specifically the new ones are 16,17,18,19)
    # We need map them to 0-4
    mapper = {k:j for k,j in zip([15,16,17,18,19], [0,1,2,3,4])}
    for anot in pseudos['annotations']:
        anot['category_id'] = mapper[anot['category_id']]

    # Check if mapper works well
    #assert set([x['category_id'] for x in pseudos['annotations']]) == set([x['category_id'] for x in gt['annotations']]) 

    final_dict = {}
    final_dict['categories'] = gt['categories']
    final_dict['images'] = pseudos['images'] + gt['images']

    # Build unique id for anotations
    anots_finales = gt['annotations']
    max = [x['id'] for x in gt['annotations']][-1] + 10  # +10 por si acaso 
    for a_ in pseudos['annotations']:
        a_['id'] = a_['id'] + max
        anots_finales.append(a_)

    final_dict['annotations'] = anots_finales


    # Save_name
    savename = args.ps.replace(".json", f"_{os.path.basename(args.gt)}.json")
    print(savename)
    with open(savename, 'w') as fp:
        json.dump(final_dict, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)