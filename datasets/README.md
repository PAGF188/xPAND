# DATASETS

## Expected dataset structure 
```
datasets/

  coco/
    annotations/
      instances_{train,val}2014.json
    {train,val}2014/

  cocosplit/
    datasplit/
      trainvalno5k.json
      5k.json
    full_box_{1,2,3,5,10,30}shot_{category}_trainval.json
    seed{1-9}/

  dataset_fsod/
    {10,30}shot_seed{0-9}
      full_box_{shot}shot_all_trainval_all.json
      full_box_{shot}shot_all_trainval_base.json
      full_box_{shot}shot_all_trainval_novel.json
    test/
      5k_base.json
      5k_novel.json
      5k.json
    trainvalno5k_base.json
    trainvalno5k_novel.json

  VOC20{07,12}/
    Annotations/
    ImageSets/
    JPEGImages/

  vocsplit/
    box_{1,2,3,5,10}shot_{category}_train.txt
    seed{1-29}/

  voc_coco/
    images/
    voc_2007_2012_trainval_all{1,2,3}.json
    voc_2007_2012_trainval_base{1,2,3}.json
    voc_2007_test_all{1,2,3}.json
    voc_2007_test_base{1,2,3}.json
    voc_2007_test_novel{1,2,3}.json
    voc_{2007,2012}_trainval_all{1,2,3}.json
    voc_{2007,2012}_trainval_all{1,2,3}_{1,2,3,5,10}shot_seed{0-29}.json
    voc_{2007,2012}_trainval_base{1,2,3}.json
    voc_{2007,2012}_trainval_novel{1,2,3}_{1,2,3,5,10}shot_seed{0-29}.json
```


## coco/
Standard COCO folder dataset

## cocosplit/
See TFA-like dataset preparation: https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md and download it from http://dl.yf.io/fs-det/datasets/. Seed0 are the .json files that are not organized in any subdirectory seed{1-9}/ and match the seed of the FSRW-like experimentation

## dataset_fsod/
To facilitate the execution we (1) add the cocosplit independent category files (full_box_{1,2,3,5,10,30}shot_{category}_trainval.json), in unify .json files. We also (2) separate 5k.json file in {base,novel,all} partitions. Run, modifying path if necessary:
```
bash run_singularity.sh
```

## VOC20{07,12}/
Standard VOC folder dataset

## vocsplit/
See TFA-like dataset preparation: https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md and download it from http://dl.yf.io/fs-det/datasets/. Seed0 are the .json files that are not organized in any subdirectory seed{1-9}/ and match the seed of the FSRW-like experimentation

## voc_coco
To facilitate the execution we convert VOC annotations to COCO-format annotations. Run, modifying path if necessary:
```
bash run_singularity.sh
```

