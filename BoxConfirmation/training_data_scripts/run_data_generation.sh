
SHARE_IADT=/data

# COCO
singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/BoxConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
    /workspace/training_data_scripts/run_coco_data.sh

# # VOC
# singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/BoxConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
#     /workspace/training_data_scripts/run_voc_data.sh
