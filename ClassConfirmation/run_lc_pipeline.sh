# Change to your path!!
SHARE_IADT=/data



# COCO ------ 
# ** IMPORTANT: It is a VitDet example. To adopt other detectors, just change the next 3 variables:
BASE_PATH=/models/VitDet/coco/mining_iter1  # Change!!
BASE_NAME=coco_instances_results  # Change!!
METHOD=VitDet  # Change!!

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    for SHOT in 10 30
    do
    singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/ClassConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_03_cu113_pytorch110.sif \
        /workspace/label_confirmation_pipeline_COCO.sh \
        $SEED $SHOT \
        $BASE_PATH/${SHOT}shot_seed${SEED} \
        $BASE_NAME \
        $METHOD
    done
done




# VOC ------
# ** IMPORTANT: It is a VitDet example. To adopt other detectors, just change the next 3 variables:
BASE_PATH=/models/VitDet/voc/mining_iter1  # Change!!
BASE_NAME=coco_instances_results  # Change!!
METHOD=VitDet  # Change!!

for SPLIT in 1 2 3    
do
    for SEED in 0 1 2 3 4
    do
        for SHOT in 1 2 3 5 10
        do
        singularity exec --nv --bind $SHARE_IADT/datasets:/datasets --bind $SHARE_IADT/models:/models --bind $HOME/xpand/ClassConfirmation:/workspace --pwd /workspace $SHARE_IADT/singularity_imgs/detectron2_03_cu113_pytorch110.sif \
            /workspace/label_confirmation_pipeline_VOC.sh $SPLIT $SEED $SHOT \
            $BASE_PATH/split_${SPLIT}/${SHOT}shot_seed${SEED} \
            $BASE_NAME \
            $METHOD
        done
    done
done

