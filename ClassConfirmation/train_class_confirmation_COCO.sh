#!/bin/bash
pip3 install transformers
# NOTE: Modify json paths according to your dataset location
# NOTE2: You can uncomment/comment the evaluation step after each epoch to get slower/faster results (train_core.py->line136).

# Base
mkdir -p /models/LC_COCO/base/
python3 classconfirmation/main.py --config configs/base_coco.yaml | tee /models/LC_COCO/base/train_log.txt


# Finetunings
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 10 30
    do
    mkdir -p /models/LC_COCO/${shot}shot_${seed}seed/
    python3 tools/config_modifier.py --config configs/finetuning_coco.yaml \
        --TRAIN_JSON /datasets/dataset_fsod/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_novel.json \
        --TEST_JSON /datasets/dataset_fsod/test/5k_novel.json \
        --OUTPUT_DIR /models/LC_COCO/${shot}shot_${seed}seed/ \
        --MODEL_WEIGHTS /models/LC_COCO/base/final_model_epoch_9.pth \
        --SUPPORT_SIZE ${shot}
    
    python3 classconfirmation/main_data_augmentation.py --config configs/finetuning_coco.yaml | tee /models/LC_COCO/${shot}shot_${seed}seed/train_log.txt
    done
done

