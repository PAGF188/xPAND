import yaml
import argparse

parser = argparse.ArgumentParser(description='Modify yaml')
parser.add_argument('--config', required=True, help='path to the config yaml')
parser.add_argument('--TRAIN_JSON', required=True, help='path to the train json ')
parser.add_argument('--OUTPUT_DIR', required=True, help='path to the output dir ')
parser.add_argument('--TEST_JSON', required=True, help='path to the test support json')
parser.add_argument('--MODEL_WEIGHTS', required=True, help='path to the model weights')
parser.add_argument('--SUPPORT_SIZE', required=True, help='path to the model weights')



if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['DATASETS']['TRAIN_JSON'] = args.TRAIN_JSON
    config['DATASETS']['SUPPORT_JSON'] = args.TRAIN_JSON
    config['OUTPUT_DIR'] = args.OUTPUT_DIR
    config['DATASETS']['TEST_JSON'] = args.TEST_JSON
    config['DATASETS']['SUPPORT_TEST_JSON'] = args.TEST_JSON
    config['MODEL_WEIGHTS'] = args.MODEL_WEIGHTS
    config['DATASETS']['SUPPORT_SIZE'] = int(args.SUPPORT_SIZE)

    with open(args.config, 'w') as f:
        yaml.dump(config, f)
