# CLASS CONFIRMATION


## Training
To train the Class Confirmation module execute:
```
bash run_train.sh
```
In this file singularity is invoked twice, once for COCO and once for VOC. Make sure you have prepared the datasets correctly.


## xPAND Pipeline
The xPAND pipeline starts with COCO format detections obtained with any detector (pseudo-labels). Make sure you have obtained them beforehand. The pipeline script is run_lc_pipeline.sh. It is for VitDet assuming that the initial detections are in /models/VitDet/coco/mining_iter1/\<shot\>shot_seed\<seed\>/coco_instances_results.json. To change to other detectors, just modify the three variables indicated in the script (BASE_PATH, BASE_NAME, METHOD).
```
run_lc_pipeline.sh
```

