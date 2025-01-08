# Singularity container
# Change to your paths!!

SHARE_IADT=/data

# COCO train
singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/ClassConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_03_cu113_pytorch110.sif \
    /workspace/train_class_confirmation_COCO.sh

# VOC train
singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/ClassConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_03_cu113_pytorch110.sif \
    /workspace/train_class_confirmation_VOC.sh