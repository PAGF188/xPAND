from dataset import COCODatasetMetaLearning, COCODatasetMetaLearningTEST
from torch.utils.data import DataLoader
import yaml
import argparse
import json
import os
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTModel
import torch
from models import LabelConfirmation, LabelConfirmation2
from train_core import Trainer
from evaluation_core import Evaluator
from importlib import import_module
import torchvision.transforms as T

parser = argparse.ArgumentParser(description='Label Confirmation training')
parser.add_argument('--config', required=True, help='path to the config yaml')

SUPPORT_BACKBONES = ["MAE", "DINO"]


def build_model(config):
    assert config['MODEL']['MODEL_FAMILY'] in SUPPORT_BACKBONES, f"{config['MODEL']['MODEL_FAMILY']} not supported in {SUPPORT_BACKBONES}"
    
    if 'MODEL_CLASS' in config['MODEL'].keys():
        model_class = getattr(import_module(config['MODEL']['MODEL_CLASS'].rsplit(".",maxsplit=1)[0]), config['MODEL']['MODEL_CLASS'].rsplit(".",maxsplit=1)[1])
    else:
        model_class = LabelConfirmation

    # Create model
    backbone = ViTModel.from_pretrained(config['MODEL']['META_ARCHITECTURE'])
    transforms = AutoImageProcessor.from_pretrained(config['MODEL']['META_ARCHITECTURE'])
    model = model_class(backbone, config['MODEL']['FINAL_TOKENS_NUMBER'], config['MODEL']['FINAL_TOKENS_AVERAGE_POOLING'])
    print(model) 
    
    return model, transforms


def main(config):

    # Check output dir exists
    if not os.path.exists(config['OUTPUT_DIR']):
        os.mkdir(config['OUTPUT_DIR'])

    # Create model
    model, transforms = build_model(config)
    model = torch.nn.DataParallel(model)
    model = model.cuda() 
    
    # Load weights if so
    if os.path.exists(config['MODEL_WEIGHTS']):
        print(f"Loading weights from {config['MODEL_WEIGHTS']}")
        model.load_state_dict(torch.load(config['MODEL_WEIGHTS']))

    # 1) Create dataloader
    dataloaders = {}
    if config['TYPE'] == 'train':
        augmentations_both = torch.nn.Sequential(
                #T.RandomResizedCrop((transforms.size['height'],transforms.size['width']), scale=(0.4,1)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.4, 0.4, 0.4, 0.2)
                )
        
        train_set = COCODatasetMetaLearning(config['DATASETS']['TRAIN_JSON'], config['DATASETS']['SUPPORT_JSON'], config['DATASETS']['DATA_FOLDER_TRAIN'], transforms, image_format=config['DATASETS']['IMG_FORMAT'], support_size=config['DATASETS']['SUPPORT_SIZE'], augmentations_query=augmentations_both, augmentations_support=augmentations_both)
        train_loader = DataLoader(train_set, batch_size=config['SOLVER']['BATCH_SIZE'], shuffle=True, num_workers=config['DATASETS']['NUM_WORKERS'])
        dataloaders['train'] = train_loader
    
    test_set = COCODatasetMetaLearningTEST(config['DATASETS']['TEST_JSON'], config['DATASETS']['SUPPORT_TEST_JSON'], config['DATASETS']['DATA_FOLDER_TEST'], transforms, image_format=config['DATASETS']['IMG_FORMAT'], support_size=config['DATASETS']['SUPPORT_SIZE'])
    test_loader = DataLoader(test_set, batch_size=config['TEST_BATCH_SIZE'], shuffle=False, num_workers=config['DATASETS']['NUM_WORKERS'])
    dataloaders['test'] = test_loader
    
    # 2) Create evaluator
    # Evaluation
    evaluator = Evaluator(config, 0)
    
    # Train loop
    if config['TYPE'] == 'train':
        trainer = Trainer(config, evaluator, 0)
        trainer.train(model, dataloaders)
    else:
        evaluator.eval_model(model, dataloaders)



if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
    main(config)



#object_crop_instance, positive_support_instance, negative_support_instance = next(iter(train_loader))
