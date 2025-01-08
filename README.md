# xPAND

## Run Pipeline
xPAND can be run on any detector as long as you have the initial detections in COCO format. The Initial Filtering and Class Confirmation steps are in the folder *ClassConfirmation/*. The Box Confirmation and Sample Selection steps are in the folder *BoxConfirmation/*. To run **one iteration** of xPAND you only have to execute:
```
1) The script xpand/ClassConfirmation/run_lc_pipeline.sh, changing the variables BASE_PATH, BASE_NAME and METHOD.
2) The script xpand/BoxConfirmation/run_bc_pipeline.sh, also changing the variables.
```

Make sure that you have previously obtained the initial detections (candidate pseudo-labels in COCO format) and have trained the ClassConfirmation (see *xpand/ClassConfirmation/README.md*) and BoxConfirmation (see *xpand/BoxConfirmation/README.md*) modules. The output will be the final pseudo-labels with which you can perform the final train of your detector.

For ease of use, we include in the detectors folder the changes to be made in the 4 detectors considered in the paper: TFA, DeFRCN, VitDet and imTED (mainly registration of fsod datasets, modification of the VOC evaluator to obtain the .json files and inclusion of ignore regions in the detectors that do not have them by default) as well as the final configuration files. Please refer to the original works to know how to execute them:

1) TFA: https://github.com/ucbdrive/few-shot-object-detection
2) DeFRCN: https://github.com/er-muyue/DeFRCN
3) VitDet: https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet
4) imTED: https://github.com/LiewFeng/imTED


## Quick test

In addition, to perform a direct execution of xPAND we attach:
1) The initial pseudo-labels MS-COCO 30-shot seed-0 for each detector
2) The Class Confirmation 30-shot seed-0 model
3) The Box Confirmation 30-shot seed-0 model
4) The singularity images

**(1), (2), (3)** can be downloaded from [here](https://nubeusc-my.sharepoint.com/:u:/g/personal/pablogarcia_fernandez_usc_es/ESHtRcVrQ1ZMgEwh7AWQpN0BF-Dlm0QJp6wnRUxctDY7yQ?e=Dj5alC). **(4)** from [here](https://nubeusc-my.sharepoint.com/:u:/g/personal/pablogarcia_fernandez_usc_es/EQKvE1n1AqxHhaVv-auN9P0BX1DYbweqJ4EpLraC9zdSEw?e=LgRnL5). With all this files you can run the full xPAND pipeline for the 4 models on MS-COCO 30-shot seed-0. Just remember to modify *xpand/ClassConfirmation/run_lc_pipeline.sh*, *xpand/BoxConfirmation/run_bc_pipeline.sh* **VARIABLES** according to the correct paths, and remove seed/shots that you will not use. 

The expected path to CC and BC weights is */models/LC_COCO/* and */models/BC_COCO/*. The expected path to initial pseudo-labels is */models/\<TFA | DeFRCN | VitDet\>/coco/mining_iter1/30shot_seed0/coco_instances_results.json* and */models/imted/coco/mining_iter1/imTED_30shot_seed0/imTED_30shot_seed0_trainval.bbox*


## To-Do List

- [ ] Release detectors folder
- [ ] Upload other pre-trained models for Class and Box Confirmation
