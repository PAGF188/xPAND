#!/bin/bash
SPLIT=$1
SEED=$2
SHOT=$3
pseudo_path=$4
pseudo_initial_name=$5
method=$6

pip3 install transformers

#Input: initial detections in COCO format. TC parameter must be False with imted
python3 pipeline/1_filter_base_fsod.py --gt /datasets/voc_coco/voc_2007_2012_trainval_all${SPLIT}.json \
    --dets ${pseudo_path}/${pseudo_initial_name}.json \
    --th 0.5 \
    --tc True \
    --voc True

python3 pipeline/2_nms_class_agnostic.py --ps ${pseudo_path}/${pseudo_initial_name}_filtrado_base2.json

python3 pipeline/3_filtrado_score_fsod.py --dets ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms.json \
    --K-min 0.2 \
    --K-max 10

python3 pipeline/4_aux_config_modifier.py --config configs/test_label_confirmation_VOC.yaml \
    --PSEUDOS_JSON ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2.json \
    --SUPPORT_JSON /datasets/voc_coco/voc_2007_trainval_novel${SPLIT}_${SHOT}shot_seed${SEED}.json \
    --OUTPUT_DIR ${pseudo_path}/LC_OUTPUT \
    --SAVE_NAME ${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC_OUTPUT.json \
    --MODEL_WEIGHTS /models/LC_VOC/${SPLIT}split_${SHOT}shot_${SEED}seed/final_model_epoch_9.pth \
    --split $SPLIT \
    --seed $SEED \
    --shot $SHOT \
    --method $method

# Execute in single GPU!!!
CUDA_VISIBLE_DEVICES=0 python3 classconfirmation/4_main_label_confirmation.py --config configs/test_label_confirmation_VOC_voc_split${SPLIT}_shot${SHOT}_seed${SEED}_${method}.yaml

python3 pipeline/5_filter_coco_LC.py --dets ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2.json \
    --filter_res ${pseudo_path}/LC_OUTPUT/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC_OUTPUT.json \
    --score_sim 0.5




