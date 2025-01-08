#!/bin/bash
pip3 install transformers
# NOTE: Modify json paths according to your dataset location
# NOTE2: You can uncomment/comment the evaluation step after each epoch to get slower/faster results (train_core.py->line136).

# Base
for split in 1 2 3
do  
    mkdir -p /models/LC_VOC/base${split}/
    python3 tools/config_modifier.py --config configs/base_voc.yaml \
            --TRAIN_JSON /datasets/voc_coco/voc_2007_2012_trainval_base${split}.json \
            --TEST_JSON /datasets/voc_coco/voc_2007_test_base${split}.json \
            --OUTPUT_DIR /models/LC_VOC/base${split}/ \
            --MODEL_WEIGHTS None \
            --SUPPORT_SIZE 10

    python3 classconfirmation/main.py --config configs/base_voc.yaml | tee /models/LC_VOC/base${split}/train_log.txt
done

# Finetunings
for split in 1 2 3
do
    for seed in 0 1 2 3 4
    do
        for shot in 1 2 3 5 10
        do
        mkdir -p /models/LC_VOC/${split}split_${shot}shot_${seed}seed/
        python3 tools/config_modifier.py --config configs/finetuning_voc.yaml \
            --TRAIN_JSON /datasets/voc_coco/voc_2007_trainval_novel${split}_${shot}shot_seed${seed}.json \
            --TEST_JSON /datasets/voc_coco/voc_2007_test_novel${split}.json \
            --OUTPUT_DIR /models/LC_VOC/${split}split_${shot}shot_${seed}seed/ \
            --MODEL_WEIGHTS /models/LC_VOC/base${split}/final_model_epoch_9.pth \
            --SUPPORT_SIZE ${shot}
        
        python3 classconfirmation/main_data_augmentation.py --config configs/finetuning_voc.yaml | tee /models/LC_VOC/${split}split_${shot}shot_${seed}seed/train_log.txt
        done
    done
done

