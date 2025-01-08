SHARE_IADT=/data # Change to your path!!
GPUS=4  # Change!!


# COCO ------ 
# ** IMPORTANT: This script continues the VitDet example started in the ClassConfirmation module. 
# To adopt other detectors, simply modify the following variables (BASE_PATH,BASE_NAME,METHOD,ITER ).
# The script works assuming the detector detections files (pseudo-labels) are in <Detector>/coco/mining_iter{iter_}/...
# (see builtin.py lines 80-96, and register your own .json files if not)
# Method must be always one of this options: ["TFA", "DeFRCN", "VitDet", "imted"]
BASE_PATH=/models/VitDet/coco/mining_iter1   # Change!!
BASE_NAME=coco_instances_results    # Change!!
METHOD=VitDet   # Change!!
ITER=1   # Change!!

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    for SHOT in 10 30
    do
    singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/BoxConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
        /workspace/box_confirmation_pipeline_COCO.sh \
        $GPUS $SEED $SHOT \
        $BASE_PATH/${SHOT}shot_seed${SEED} \
        $BASE_NAME \
        $METHOD \
        1
    done
done




# VOC ------
# ** IMPORTANT: This script continues the VitDet example started in the ClassConfirmation module. 
# To adopt other detectors, simply modify the following variables (BASE_PATH,BASE_NAME,METHOD,ITER ).
# The script works assuming the detector detections files (pseudo-labels) are in <Detector>/voc/mining_iter{iter_}/...
# (see builtin.py lines 100-120, and register your own .json files if not)
# Method must be always one of this options: ["TFA", "DeFRCN", "VitDet", "imted"]
BASE_PATH=/models/VitDet/voc/mining_iter1  # Change!!
BASE_NAME=coco_instances_results   # Change!!
METHOD=VitDet   # Change!!
ITER=1   # Change!!

for SPLIT in 1 2 3    
do
    for SEED in 0 1 2 3 4
    do
        for SHOT in 1 2 3 5 10
        do
        singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/BoxConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_06_7_cu113_pytorch110.sif \
            /workspace/box_confirmation_pipeline_VOC.sh \
            $GPUS $SEED $SHOT \
            $BASE_PATH/split_${SPLIT}/${SHOT}shot_seed${SEED} \
            $BASE_NAME \
            $METHOD \
            $ITER \
            $SPLIT
        done
    done
done

