from dataset import COCODatasetBase
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTModel
import torch
import numpy as np
import json


parser = argparse.ArgumentParser(description='Extract MAE or DINO features')
parser.add_argument('--config', required=True, help='path to the config yaml')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORT_BACKBONES = ["MAE", "DINO"]



def save_coco(coco, anns_all, save_name):
    save_dict = {}
    for k in coco.dataset.keys():
        if k != 'annotations' and k!= 'images':
            save_dict[k] = coco.dataset[k]
    # Añadir anotaciones
    save_dict['annotations'] = anns_all
    # Añadir imagenes
    anotaciones_filtradas_imgs_ids = list(set([a['image_id'] for a in anns_all]))
    anotaciones_filtradas_imgs = coco.loadImgs(anotaciones_filtradas_imgs_ids)
    save_dict['images'] = anotaciones_filtradas_imgs

    print(save_name)
    with open(save_name, 'w') as fp:
        json.dump(save_dict, fp, indent=4, sort_keys=True)

    return save_name




def build_model(config):
    assert config['MODEL']['MODEL_FAMILY'] in SUPPORT_BACKBONES, f"{config['MODEL']['MODEL_FAMILY']} not supported in {SUPPORT_BACKBONES}"
    
    # Create model
    model = ViTModel.from_pretrained(config['MODEL']['META_ARCHITECTURE'])
    transforms = AutoImageProcessor.from_pretrained(config['MODEL']['META_ARCHITECTURE'])
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)  
    
    return model, transforms


def print_metric(dict):
    for k in dict.keys():
        print(f"{k}: {dict[k]}")
    print(f"Mean class: {np.mean(np.array(list(dict.values())))}")


def main(config):
    
    ### Create model
    model, transforms = build_model(config)
    
    # Create dataloader
    train_set = COCODatasetBase(config['DATASETS']['DETS_JSON'], config['DATASETS']['DATA_FOLDER'], transforms, image_format=config['DATASETS']['IMG_FORMAT'])
    train_loader = DataLoader(train_set, batch_size=config['TEST_BATCH_SIZE'], shuffle=True, num_workers=config['DATASETS']['NUM_WORKERS'])

    # Extract and save features (and object id's)
    CLASSES_ = train_set.get_classes()
    features_class = {x:{} for x in CLASSES_}
    model.eval()
    with torch.no_grad():
        iterator = tqdm(train_loader)
        for object_crop_instance in iterator:
            features = model(object_crop_instance['object_crop'])
            
            if config['MODEL']['FINAL_TOKENS_AVERAGE_POOLING']:
               for f_,c_, id_ in zip(features.last_hidden_state, object_crop_instance['category_id'], object_crop_instance['anot_id']):
                    features_class[int(c_)][int(id_)] = torch.mean(f_, dim=0).cpu().numpy().tolist()
            else:
                for f_,c_, id_ in zip(features.last_hidden_state[:,0], object_crop_instance['category_id'], object_crop_instance['anot_id']):
                    features_class[int(c_)][int(id_)] = f_.cpu().numpy().tolist()

        
    # Save new annotations to new json
    save_name = config['DATASETS']['DETS_JSON'].replace(".json", f"extracted_features{config['MODEL']['MODEL_FAMILY']}.json")
    print(save_name)
    with open(save_name, 'w') as fp:
        json.dump(features_class, fp, indent=4, sort_keys=True)



    

            



if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)


