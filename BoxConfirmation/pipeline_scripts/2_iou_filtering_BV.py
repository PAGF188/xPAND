import argparse
from pycocotools.coco import COCO
from tqdm.auto import tqdm
import json


parser = argparse.ArgumentParser(description='')
parser.add_argument('--pseudos', required=True, help='path to the pseudo data')
parser.add_argument('--box_ver_output', required=True, help='path to the box verification output')
parser.add_argument('--th', required=True, help='Threshold predicted box')


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
    try:
        coco_dt = COCO(args.pseudos)
    except:
        print(f"{args.pseudos} has not a valid format")

    box_ver_output = json.load(open(args.box_ver_output))

    # 1) Get the keep ids
    keep_ids = [x['id'] for x in box_ver_output if x['predicted_iou'] >= float(args.th)] 

    # 2) Get annotations of this ids
    anns_filtered = coco_dt.loadAnns(keep_ids)

    # 3) Save results
    save_name = args.pseudos.replace(".json", f"_box_verification{args.th}.json")
    save_coco(coco_dt, anns_filtered, save_name)
     


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)