from dataset import COCOPseudoLabelMining, COCOSupportDataset
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


parser = argparse.ArgumentParser(description='Label Confirmation')
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


def precompute_support(config, model, transforms):
    
    print("Precomputing support")
    # 1) Create dataloader
    support_set = COCOSupportDataset(config['DATASETS']['SUPPORT_JSON'], config['DATASETS']['DATA_FOLDER_SUPPORT'], transforms, image_format=config['DATASETS']['IMG_FORMAT'])
    support_dataloader = DataLoader(support_set, batch_size=1, shuffle=False, num_workers=8)
    support_grouped_class = {key:[] for key in support_set.get_classes()}

    with torch.set_grad_enabled(False):
        for support_instance in tqdm(support_dataloader):
            s = support_instance['object_crop'].cuda()
            out = model.module.backbone(s[0]).last_hidden_state
            
            if config['MODEL']['FINAL_TOKENS_AVERAGE_POOLING']:
                out = torch.mean(out[:,1:,:], dim=1)
            else:
                out = out[:,0,:]

            out = torch.mean(out, dim=0)
            support_grouped_class[int(support_instance['category_id'])] = out.cpu().numpy()
    
    return support_grouped_class


def main(config):

    # Check output dir exists
    if not os.path.exists(config['OUTPUT_DIR']):
        os.mkdir(config['OUTPUT_DIR'])

    # Create model
    model, transforms = build_model(config)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    # Load weights if so
    if os.path.exists(config['MODEL_WEIGHTS']):
        print(f"Loading weights from {config['MODEL_WEIGHTS']}")
        model.load_state_dict(torch.load(config['MODEL_WEIGHTS']))


    ################################################
    ############ 1.-PRECOMPUTE SUPPORT
    support_precomputed = precompute_support(config, model, transforms)


    ################################################
    ############ 1.-LABEL CONFIRMATION STEP
    # 1) Create dataloader
    pseudo_set = COCOPseudoLabelMining(config['DATASETS']['PSEUDOS_JSON'], support_precomputed, config['DATASETS']['DATA_FOLDER_PSEUDOS'], transforms, image_format=config['DATASETS']['IMG_FORMAT'])
    pseudo_loader = DataLoader(pseudo_set, batch_size=config['TEST_BATCH_SIZE'], shuffle=False, num_workers=config['DATASETS']['NUM_WORKERS'])
    
    # 2) Create evaluator
    # Evaluation
    evaluator = Evaluator(config, 0)
    
    # 3) Label Confirmation
    evaluator.label_confirmation(model, pseudo_loader, config['SAVE_NAME'])



if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
    main(config)



#object_crop_instance, positive_support_instance, negative_support_instance = next(iter(train_loader))
