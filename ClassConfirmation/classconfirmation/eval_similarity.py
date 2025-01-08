from dataset import COCODatasetBase
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTMAEModel, ViTModel, ViTImageProcessor, ViTFeatureExtractor
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

parser = argparse.ArgumentParser(description='Eval similarity MAE/DINO features')
parser.add_argument('--config', required=True, help='path to the config yaml')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORT_BACKBONES = ["MAE", "DINO"]



def build_model(config):
    assert config['MODEL']['MODEL_FAMILY'] in SUPPORT_BACKBONES, f"{config['MODEL']['MODEL_FAMILY']} not supported in {SUPPORT_BACKBONES}"
    
    # Create model
    model = ViTModel.from_pretrained(config['MODEL']['META_ARCHITECTURE'])
    transforms = AutoImageProcessor.from_pretrained(config['MODEL']['META_ARCHITECTURE'])
    #print(model)
    model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)  
    
    return model, transforms


def print_metric(dict):
    for k in dict.keys():
        #print(f"{k}: {dict[k]}")
        print(f"{dict[k]}")

    print(f"Mean class: {np.mean(np.array(list(dict.values())))}")

def main(config):
    
    ### Create model
    model, transforms = build_model(config)
    
    # Create dataloader
    train_set = COCODatasetBase(config['DATASETS']['TRAIN_JSON'], config['DATASETS']['DATA_FOLDER'], transforms, image_format=config['DATASETS']['IMG_FORMAT'])
    train_loader = DataLoader(train_set, batch_size=config['SOLVER']['BATCH_SIZE'], shuffle=True, num_workers=8)

    # Extract and save features
    CLASSES_ = train_set.get_classes()
    features_class = {x:[] for x in CLASSES_}
    model.eval()
    with torch.no_grad():
        iterator = tqdm(train_loader)
        for object_crop_instance in iterator:
            features = model(object_crop_instance['object_crop'])

            if config['MODEL']['FINAL_TOKENS_AVERAGE_POOLING']:
               for f_,c_ in zip(features.last_hidden_state, object_crop_instance['category_id']):
                   features_class[int(c_)].append(torch.mean(f_[1:,:], dim=0))
            else:
                for f_,c_ in zip(features.last_hidden_state[:,0], object_crop_instance['category_id']):
                    features_class[int(c_)].append(f_)

    
    ### Compute similarity intra-class (mean of max)
    mean_of_means = {x:None for x in CLASSES_}  # Mean of mean similarities per class

    for k_ in CLASSES_:
        fueatures_k_class = torch.stack(features_class[k_])
        fueatures_k_class.sub(torch.mean(fueatures_k_class))
        fueatures_k_class = fueatures_k_class.cpu().detach().numpy()
        cosine_matrix_sim = cosine_similarity(fueatures_k_class)

        # Mean of similarities. Ignoring diagonal elements
        mean_of_means[k_] = np.mean((cosine_matrix_sim.sum(0)-np.diag(cosine_matrix_sim))/(cosine_matrix_sim.shape[0]-1))

    # Print results
    print("\nMean of means:")
    print_metric(mean_of_means)


    

            



if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    

    main(config)


