#!/bin/bash
seed=$1
shot=$2
pseudo_path=$3
pseudo_initial_name=$4
method=$5

pip3 install transformers

#Input: initial detections in COCO format. TC parameter must be False with imted
python3 pipeline/1_filter_base_fsod.py --gt /datasets/cocosplit/datasplit/trainvalno5k.json \
    --dets ${pseudo_path}/${pseudo_initial_name}.json \
    --th 0.5 \
    --tc True \
    --voc False

python3 pipeline/2_nms_class_agnostic.py --ps ${pseudo_path}/${pseudo_initial_name}_filtrado_base2.json

# ** The 0.5 score filtering is in the box confirmation pipeline script !!!
python3 pipeline/3_filtrado_score_fsod.py --dets ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms.json \
    --K-min 0.2 \
    --K-max 10

python3 pipeline/4_aux_config_modifier.py --config configs/test_label_confirmation_COCO.yaml \
    --PSEUDOS_JSON ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2.json \
    --SUPPORT_JSON /datasets/dataset_fsod/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_novel.json \
    --OUTPUT_DIR ${pseudo_path}/LC_OUTPUT \
    --SAVE_NAME ${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC_OUTPUT.json \
    --MODEL_WEIGHTS /models/LC_coco_finetunings/${shot}shot_seed${seed}/final_model_epoch_9.pth \
    --shot $shot \
    --seed $seed \
    --method $method

# Execute in single GPU!!!
CUDA_VISIBLE_DEVICES=0 python3 classconfirmation/4_main_label_confirmation.py --config configs/test_label_confirmation_COCO_coco_shot${shot}_seed${seed}_${method}.yaml

python3 pipeline/5_filter_coco_LC.py --dets  ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2.json \
    --filter_res ${pseudo_path}/LC_OUTPUT/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC_OUTPUT.json \
    --score_sim 0.5

