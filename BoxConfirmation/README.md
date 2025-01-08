# BOX CONFIRMATION


## COCO Training Data Preparation
To prepare the data for training the Box Confirmation module:

1) Base data is extracted from the initial boxes of a detector. Run the inference of your base detector on trainvalno5k.json and paste the result in /datasets/BC_COCO/coco_instances_results.json. Then execute:

```
python3 training_data_scripts/selection_train_test_examples.py \
    --gt /datasets/dataset_fsod/trainvalno5k.json
    --dets <initial_detections>
```

Register the result in xpand/BoxConfirmation/data/builtin.py (example on lines 44,45)

2) Finetuning data is obtained with generate_sintetic_finetuning_data.py. Run on each shot/seed:
```
python3 generate_sintetic_finetuning_data.py \
    --gt /datasets/dataset_fsod/<shot>shot_seed<seed>/full_box_<shot>shot_all_trainval_novel.json
    --output /datasets/BC_COCO/

python3 training_data_scripts/selection_train_test_examples.py \
    --dets </datasets/BC_COCO/full_box_<shot>shot_all_trainval_novel_sint_boxes.json>
```
Register the result in xpand/BoxConfirmation/data/builtin.py

**This process is made in script xpand/BoxConfirmation/training_data_scripts/run_data_generation.sh. You only have to paste the base detection json file in /datasets/BC_COCO/coco_instances_results.json, and register the .json in builtin.py (already registered if you use the same paths)**

## VOC Training Data Preparation
The same for VOC. You only have to paste the initial detections on /datasets/BC_VOC/base\{split\}/coco_instances_results.json (one folder for each split). Then run xpand/BoxConfirmation/training_data_scripts/run_data_generation.sh. Make sure that the resulting .json files are registered in builtin.py 




## COCO and VOC Training
To train the Class Confirmation module execute:
```
bash run_train_all.sh
```
This file is the execution process for both COCO and VOC. Change lines 2-3 if necessary. Make sure you have prepared the data correctly.


## xPAND Pipeline
The xPAND pipeline continues with the pseudo-labels obtained from ClassConfirmation output (see xpand/ClassConfirmation/README.md and make sure you have run it correctly!). The pipeline script is xpand/BoxConfirmation/run_bc_pipeline.sh. Its prepared for VitDet. To run other detectors, just modify the three variables indicated in the script (BASE_PATH, BASE_NAME, METHOD, ITER).
```
bash run_bc_pipeline.sh
```

# Stop Criterion
The stop criterion is in stop_criterion_vfast.py. You have to call that file with the pseudo-labels of the Box Confirmation module between iter n, and iter n+1:
```
python3 stop_criterion_vfast.py \
    --dets1 <Box Confirmation pseudo-labels at iteration n>
    --dets2 <Box Confirmation pseudo-labels at iteration n+1>

# Example:
python3 stop_criterion_vfast.py \
--dets1 /models/VitDet/coco/mining_iter1/10shot_seed0/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json \
--dets2 /models/VitDet/coco/mining_iter2/10shot_seed0/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json \
--dataset COCO
```
