import json
from tqdm import tqdm
import argparse
from pycocotools.coco import COCO


parser = argparse.ArgumentParser(description='Add sim score to coco annotations')
parser.add_argument('--dets', required=True, help='path to the pseudo data')
parser.add_argument('--filter_res', required=True, help='path to the filtered LC results')


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



def build_dict_sim(anots_sim_pos, anots_sim_neg):
    anots_sim_totales = {}

    for x in anots_sim_pos:
        anots_sim_totales[x['id']] = x['score_sim']

    for x in anots_sim_neg:
        anots_sim_totales[x['id']] = x['score_sim']

    return anots_sim_totales

def main(args):
    coco_dets = COCO(args.dets)
    coco_dets_anots = coco_dets.dataset['annotations']

    # SCORE SIM
    anots_sim = json.load(open(args.filter_res))
    anots_sim_pos = anots_sim['keep']
    anots_sim_neg = anots_sim['remove']
    anots_sim_totales = build_dict_sim(anots_sim_pos, anots_sim_neg)

    for anot in tqdm(coco_dets_anots):
        anot['score_sim'] = anots_sim_totales[anot['id']]
    
    save_name = args.dets.replace(".json", f"_LC_score_sim.json")
    save_coco(coco_dets, coco_dets_anots, save_name)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
