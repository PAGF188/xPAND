# Note: Change to your path. Under $P must be the datasets/ folder
P=/

# COCO dataset preparation
singularity exec --nv --bind /data/datasets:/datasets --bind $HOME/xpand/datasets:/workspace --pwd /workspace /data/singularity_imgs/detectron2_03_cu113_pytorch110.sif \
    /workspace/prepare_coco_fsod.sh $P

# VOC dataset preparation
singularity exec --nv --bind /data/datasets:/datasets --bind $HOME/xpand/datasets:/workspace --pwd /workspace /data/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
    /workspace/prepare_voc_fsod.sh $P