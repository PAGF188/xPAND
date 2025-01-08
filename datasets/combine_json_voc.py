import argparse
import json
from pycocotools.coco import COCO
import random
import numpy as np


parser = argparse.ArgumentParser(description='')
parser.add_argument('--d1', required=True, help='path1 ')
parser.add_argument('--d2', required=True, help='path2 ')
parser.add_argument('--output', required=True, help='Output file')

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
    anots1 = json.load(open(args.d1))
    anots2 = json.load(open(args.d2))

    assert anots1['categories'] == anots2['categories'] 

    final_dict = {}
    final_dict['info'] = anots1['info']
    final_dict['categories'] = anots1['categories']
    final_dict['licenses'] = anots1['licenses']
    final_dict['images'] = anots1['images'] + anots2['images']

    # Images and annotations ids must be unique!!
    anots_finales = anots1['annotations']
    max = [x['id'] for x in anots1['annotations']][-1] + 10  # +10 just in case 
    for a_ in anots2['annotations']:
        a_['id'] = a_['id'] + max
        anots_finales.append(a_)

    final_dict['annotations'] = anots_finales


    # Save_name
    with open(args.output, 'w') as fp:
        json.dump(final_dict, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)