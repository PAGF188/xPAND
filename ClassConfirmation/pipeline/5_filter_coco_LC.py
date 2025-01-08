import argparse
import json
from pycocotools.coco import COCO
from tqdm.auto import tqdm


parser = argparse.ArgumentParser(description='Filtering pseudos not passing LC')
parser.add_argument('--dets', required=True, help='path to the pseudo data')
parser.add_argument('--filter_res', required=True, help='path to the filtered LC results')
parser.add_argument('--score_sim', required=False, default=0.5, help='Similarity threshold. Default 0.5 ')


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


def main(args):
    with open(args.filter_res) as f:
        filter_res_lc = json.load(f)
    coco_pseudos = COCO(args.dets)

    # 1) Get the keep ids
    keep_ids = [x['id'] for x in filter_res_lc['keep'] if x['score_sim'] >= float(args.score_sim)] 

    # 2) Get annotations of this ids
    anns_filtered = coco_pseudos.loadAnns(keep_ids)

    # 3) Save results
    save_name = args.dets.replace(".json", f"_LC{args.score_sim}.json")
    save_coco(coco_pseudos, anns_filtered, save_name)



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
