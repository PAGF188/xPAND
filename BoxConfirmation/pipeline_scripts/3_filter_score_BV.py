import argparse
import json
from pycocotools.coco import COCO


parser = argparse.ArgumentParser(description='Compute IoU')
parser.add_argument('--dets', required=True, help='path to the dets data')
parser.add_argument('--score_th', required=True, help='path to the dets data')


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
    anots = json.load(open(args.dets))
    anots = anots['annotations']
    th = float(args.score_th)
    anots_filt = [x for x in anots if float(x['score']) >= th] 

    save_name = args.dets.replace(".json", f"_score{args.score_th}.json")
    save_coco(COCO(args.dets), anots_filt, save_name)
   



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)