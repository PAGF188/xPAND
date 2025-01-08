#!/bin/bash
FOLDER=$1
PERCENTAJE=$2

#pip3 install transformers
# NOTE: Modify json paths according to your dataset location
# NOTE2: You can uncomment/comment the evaluation step after each epoch to get slower/faster results (train_core.py->line136).

# Base
mkdir -p /models/LC_COCO_SSOD/${FOLDER}folder_${PERCENTAJE}per/
python3 classconfirmation/main_SSOD.py --config configs/SSOD/SSOD_1@1.yaml | tee /models/LC_COCO_SSOD/${FOLDER}folder_${PERCENTAJE}per/train_log.txt