#/bin/bash

P=$1 # Note: Change to your path. Under $P must be the datasets/ folder

#1) Create seed0 folder
mkdir ${P}datasets/cocosplit/seed0
cp ${P}datasets/cocosplit/full_box*.json ${P}datasets/cocosplit/seed0/


#2) Prepare fsod_coco (just combining previous separated jsons). Modify paths if needed
mkdir ${P}datasets/dataset_fsod/
for shot in 10 30
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        mkdir ${P}datasets/dataset_fsod/${shot}shot_seed${seed}
    done
done
python3 combine_json.py --path ${P}datasets/cocosplit/ --output ${P}datasets/dataset_fsod/



#3) Split novel/base parts. Modify paths if needed
for shot in 10 30
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        python3 base_novel_split_annotations_coco.py --gt ${P}datasets/dataset_fsod/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_all.json
        mv ${P}datasets/dataset_fsod/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_all_base.json ${P}datasets/dataset_fsod/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_base.json
        mv ${P}datasets/dataset_fsod/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_all_novel.json ${P}datasets/dataset_fsod/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_novel.json 
    done
done


#4) Copy test. Modify paths if needed
mkdir ${P}datasets/dataset_fsod/test
cp ${P}datasets/cocosplit/datasplit/5k.json ${P}datasets/dataset_fsod/test/
python3 base_novel_split_annotations_coco.py --gt ${P}datasets/dataset_fsod/test/5k.json


#5) Copy trainval. Modify paths if needed
cp ${P}datasets/cocosplit/datasplit/trainvalno5k.json ${P}datasets/dataset_fsod/
python3 base_novel_split_annotations_coco.py --gt ${P}datasets/dataset_fsod/trainvalno5k.json




