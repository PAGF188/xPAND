SHARE_IADT=/data
FOLDER=$1
PERCENTAJE=$2

# COCO train
singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/ClassConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_03_cu113_pytorch110.sif \
    /workspace/train_class_confirmation_COCO_SSOD.sh $FOLDER $PERCENTAJE
