import numpy as np
from tabulate import tabulate
import contextlib
import io
from pycocotools.coco import COCO
from collections import defaultdict
from tqdm.auto import tqdm
import argparse


parser = argparse.ArgumentParser(description='Stop criterion')

parser.add_argument('--dets1', required=True, help='Path to the dets annotation data after BoxConfirmation, iteration n')
parser.add_argument('--dets2', required=True, help='Path to the dets annotation data after BoxConfirmation, iteration n+1')

def print_table(table):
    for c in table:
        print(f"{c[0]}: {c[1]}")
    print()

def count(filename_dt):
    # [[clase, num]]
    count_ = []
    try:
        coco_dt = COCO(filename_dt)
    except:
        print("Stop: True\n")
        exit()
    clases = set([x['category_id'] for x in coco_dt.dataset['annotations']])
    clases_names = [coco_dt.cats[x]['name'] for x in clases]

    for c, cn in zip(clases, clases_names):
        n = [x for x in coco_dt.dataset['annotations'] if x['category_id']==c]
        count_.append([cn, len(n)])

    #print_table(count_)
    return count_


# Si el X% (25%) de las categoria disminuyen su numero de etiqeutas. 5 clases for voc
def criterio2(res_clase_iter1, res_clase_iter2, por=0.25):
    res_clase_iter1_names = [x[0] for x in res_clase_iter1]
    res_clase_iter1_nums = [x[1] for x in res_clase_iter1]
    
    res_clase_iter2_names = [x[0] for x in res_clase_iter2]  
    res_clase_iter2_nums = [x[1] for x in res_clase_iter2] 

    if len(res_clase_iter2_names) < len(res_clase_iter1_names):
        print(0.0)
        return True
    
    res_iter2_all = []   # Same order than res_clase_iter1_names (and res_clase_iter1_names)

    for name_iter1 in res_clase_iter1_names:
        try:
            res_clase_iter2_index = res_clase_iter2_names.index(name_iter1)
            res_iter2_all.append(res_clase_iter2_nums[res_clase_iter2_index])
        except:
            res_iter2_all.append(0)

    # We stop if res_iter2_all is less than res_class_iter1_nums in more than 25% of the classes.
    diff = np.array(res_iter2_all) - np.array(res_clase_iter1_nums)
    diff = diff / res_clase_iter1_nums
    #print(diff)
    print(f"{np.median(diff)}")
    return np.median(diff) <= por


def main(args):

    res_clase_iter1 = count(args.dets1)
    res_clase_iter2 = count(args.dets2)
        
    crit = criterio2(res_clase_iter1, res_clase_iter2)
    print(f"Stop: {crit}\n")
    
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)