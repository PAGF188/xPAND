import yaml
import argparse

parser = argparse.ArgumentParser(description='Modify yaml')
parser.add_argument('--config', required=True, help='path to the config yaml')
parser.add_argument('--PSEUDOS_JSON', required=True, help='path to the train json ')
parser.add_argument('--SUPPORT_JSON', required=True, help='path to the train json support')
parser.add_argument('--OUTPUT_DIR', required=True, help='path to the output dir ')
parser.add_argument('--SAVE_NAME', required=True, help='Save name ')
parser.add_argument('--MODEL_WEIGHTS', required=True, help='Model weights ')
parser.add_argument('--method', required=True, help='Name of the initial detector ')

parser.add_argument('--split', required=False, default=None, help='split ')
parser.add_argument('--shot', required=True, help='shot ')
parser.add_argument('--seed', required=True, help='seed ')



if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['DATASETS']['PSEUDOS_JSON'] = args.PSEUDOS_JSON
    config['DATASETS']['SUPPORT_JSON'] = args.SUPPORT_JSON
    config['OUTPUT_DIR'] = args.OUTPUT_DIR
    config['SAVE_NAME'] = args.SAVE_NAME
    config['MODEL_WEIGHTS'] = args.MODEL_WEIGHTS

    if args.split is not None:
        split = int(args.split)
        seed = int(args.seed)
        shot = int(args.shot)
        save_name = args.config.replace(".yaml", f"_voc_split{split}_shot{shot}_seed{seed}_{args.method}.yaml")
    else:
        seed = int(args.seed)
        shot = int(args.shot)
        save_name = args.config.replace(".yaml", f"_coco_shot{shot}_seed{seed}_{args.method}.yaml")

    with open(save_name, 'w') as f:
        yaml.dump(config, f)
