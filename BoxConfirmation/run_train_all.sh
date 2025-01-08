
SHARE_IADT=/data   # Modify!!
GPUS=4  # Modify!!            

# COCO ----------------------------------------------------------------------
mkdir $SHARE_IADT/models/BC_COCO/

# Base training
CONFIG_FILE=configs/box_ref.py
TRAIN_NAME=coco_base_train
TEST_NAME=coco_base_test
OUTPUT_DIR=/models/BC_COCO/base/
INIT_WEIGHTS="None"
singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/BoxConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
    /workspace/run_train_single.sh $GPUS $CONFIG_FILE $TRAIN_NAME $TEST_NAME $OUTPUT_DIR $INIT_WEIGHTS

# Finetunings
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 10 30
    do
        CONFIG_FILE=configs/box_ref_finetuning.py
        TRAIN_NAME=full_box_${shot}shot_seed${seed}_all_trainval_novel_sint_boxes_box_selected_train
        TEST_NAME=full_box_${shot}shot_seed${seed}_all_trainval_novel_sint_boxes_box_selected_test
        OUTPUT_DIR=/models/BC_COCO/${shot}shot_seed${seed}/
        INIT_WEIGHTS=/models/BC_COCO/base/model_final.pth
        singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/BoxConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
            /workspace/run_train_single.sh $GPUS $CONFIG_FILE $TRAIN_NAME $TEST_NAME $OUTPUT_DIR $INIT_WEIGHTS
    done
done



# # VOC ----------------------------------------------------------------------
mkdir $SHARE_IADT/models/BC_VOC/

# Base
for split in 1 2 3
do
    CONFIG_FILE=configs/box_ref_voc.py
    TRAIN_NAME=voc_base${split}_train
    TEST_NAME=voc_base${split}_test
    OUTPUT_DIR=/models/BC_VOC/base${split}
    INIT_WEIGHTS="None"
    singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/BoxConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
            /workspace/run_train_single.sh $GPUS $CONFIG_FILE $TRAIN_NAME $TEST_NAME $OUTPUT_DIR $INIT_WEIGHTS
done

# # Finetunings
for split in 1 2 3
do
    for seed in 0 1 2 3 4
    do
        for shot in 1 2 3 5 10
        do
            CONFIG_FILE=configs/box_ref_finetuning.py
            TRAIN_NAME=voc_2007_trainval_novel${split}_${shot}shot_seed${seed}_sint_boxes_box_selected_train
            TEST_NAME=voc_2007_trainval_novel${split}_${shot}shot_seed${seed}_sint_boxes_box_selected_test
            OUTPUT_DIR=/models/BC_VOC/split${split}/${shot}shot_seed${seed}/
            INIT_WEIGHTS=/models/BC_VOC/base${split}/model_final.pth
            singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/BoxConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
                /workspace/run_train_single.sh $GPUS $CONFIG_FILE $TRAIN_NAME $TEST_NAME $OUTPUT_DIR $INIT_WEIGHTS
        done
    done
done



