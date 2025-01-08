from pycocotools.coco import COCO
import argparse
from tqdm import tqdm
import os
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.structures.boxes import BoxMode

UNSEEN_IDS = [ 1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
IMGS_IDS_FIXED = []
#IMGS_IDS_FIXED = [412908, 186558, 536429, 279334, 108722, 468867, 8644, 45181, 397645, 389109, 11613, 5557, 203054, 253785, 340258, 376065, 551169, 425341, 529117, 492156, 46731, 187952, 408418, 26445, 6873, 137396, 525264, 10643, 313757, 301045, 206749, 147753, 234236, 347313, 191690, 381315, 368978, 474446, 107672, 381021, 303607, 553669, 309382, 136596, 114108, 168852, 358252, 484960, 8944, 250344, 144436, 415933, 331324, 515701, 149623, 143370, 48908, 578391, 426118, 335585, 62770, 232116, 232091, 466020, 27433, 186753, 417802, 257021, 382617, 172924, 185917, 421996, 320670, 557308, 90570, 34539, 354278, 330954, 366733, 1611, 101749, 181322, 498079, 554860, 405851, 224395, 426831, 129285, 210731, 353012, 188599, 493544, 286794, 402221, 64899, 127899, 232692, 230268, 53088, 412632]

parser = argparse.ArgumentParser(description='Basic visualicer')
parser.add_argument('--dets', help="pseudos", required=True,  type=str)
parser.add_argument('--gt', required=False, help='path to the gt data', default="datasets/cocosplit/datasplit/trainvalno5k.json")
parser.add_argument('--output', help="Output", required=True,  type=str)


def main(args):

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    gt_data = COCO(args.gt)
    try:
        pseudos_data = COCO(args.dets)
    except:
        pseudos_data = gt_data.loadRes(args.dets)
    
    img_ids = pseudos_data.getImgIds()
    if len(IMGS_IDS_FIXED) != 0:
        selected_img_ids =  IMGS_IDS_FIXED
    else:
        selected_img_ids = random.sample(img_ids, 100)

    print("Imagenes ids: \n", selected_img_ids)

    for img_id in tqdm(selected_img_ids):
        # Load image
        img_info = pseudos_data.loadImgs([img_id])[0]
        # Load annotations ids
        pseudos_ids = pseudos_data.getAnnIds(imgIds=[img_id], iscrowd=None)
        # Load gt ids
        real_gt_ids = gt_data.getAnnIds(imgIds=[img_id], iscrowd=None)

        # Load ids annotations
        pseudos_ans = pseudos_data.loadAnns(pseudos_ids)
        real_gt_ans = gt_data.loadAnns(real_gt_ids)

        img_name = "datasets/coco/trainval2014/" + img_info['file_name']
        im = cv2.imread(img_name)
        
        v = Visualizer(
            im[:,:,::-1]
        )
        
        for an_ in pseudos_ans:
            if an_['category_id'] in UNSEEN_IDS and an_['ignore_qe']!=1:
                box = BoxMode.convert(an_['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                v.draw_box(box, edge_color='g')
                v.draw_text(f"{an_['category_id']}", (box[0], box[1]), color='g')


        for an_ in real_gt_ans:
            if an_['category_id'] in UNSEEN_IDS:
                box = BoxMode.convert(an_['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                v.draw_box(box, edge_color='r')
                v.draw_text(f"{an_['category_id']}", (box[0], box[1]), color='r')

        v = v.get_output()
        img =  v.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(args.output, img_info['file_name']), img)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    
    main(args)