# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import _get_builtin_metadata
from .meta_coco import register_coco_instances

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    # Base trainings COCO
    "coco_base_train": ("coco/trainval2014", "BC_COCO/coco_instances_results_box_selected_train.json"),
    "coco_base_test": ("coco/trainval2014", "BC_COCO/coco_instances_results_box_selected_test.json"),
}

_PREDEFINED_SPLITS_VOC = {}
_PREDEFINED_SPLITS_VOC["voc"] = {
    # Base trainings VOC
    "voc_base1_train": ("voc_coco/images", "BC_VOC/base1/coco_instances_results_box_selected_train.json", 1, "base"),
    "voc_base1_test": ("voc_coco/images", "BC_VOC/base1/coco_instances_results_box_selected_test.json", 1, "base"),

    "voc_base2_train": ("voc_coco/images", "BC_VOC/base2/coco_instances_results_box_selected_train.json", 2, "base"),
    "voc_base2_test": ("voc_coco/images", "BC_VOC/base2/coco_instances_results_box_selected_test.json", 2, "base"),

    "voc_base3_train": ("voc_coco/images", "BC_VOC/base3/coco_instances_results_box_selected_train.json", 3, "base"),
    "voc_base3_test": ("voc_coco/images", "BC_VOC/base3/coco_instances_results_box_selected_test.json", 3, "base"),

}

# VOC finetunings
for split in [1,2,3]:
    for seed in [0,1,2,3,4]:
        for shot in [1,2,3,5,10]:
            for prefix in ['train', 'test']:
                name = f"voc_2007_trainval_novel{split}_{shot}shot_seed{seed}_sint_boxes_box_selected_{prefix}"
                json_dir = f"BC_VOC/finetunings/voc_2007_trainval_novel{split}_{shot}shot_seed{seed}_sint_boxes_box_selected_{prefix}.json" 
                _PREDEFINED_SPLITS_VOC["voc"][name] = ("voc_coco/images", json_dir, int(split), "novel")


# COCO finetunings
for shot in [10, 30]:
    for seed in range(10):
        for prefix in ['train', 'test']:
            name = f"full_box_{shot}shot_seed{seed}_all_trainval_novel_sint_boxes_box_selected_{prefix}"
            json_dir = f"BC_COCO/{shot}shot_seed{seed}/full_box_{shot}shot_all_trainval_novel_sint_boxes_box_selected_{prefix}.json" 
            _PREDEFINED_SPLITS_COCO["coco"][name] = ("coco/trainval2014", json_dir)




## =====================================================================================================================
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TESTS xPAND pipeline
## =====================================================================================================================

# COCO PSEUDOS FROM ITER1-4
# NOTE: CHANGE IF YOU ARE USING DIFFERENT PATHS!!!!
for prefix in ["TFA", "DeFRCN", "VitDet", "imted"]:
    for iter_ in [1, 2, 3, 4]:
        for shot in [10, 30]:
            for seed in range(10):
                name = "{}_coco_pseudos_{}shot_seed{}_iter{}".format(prefix, shot, seed, iter_)
                if prefix == "VitDet":
                    json_dir = f"VitDet/coco/mining_iter{iter_}/{shot}shot_seed{seed}/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_dummy_iou.json"
                elif prefix == "TFA":
                    json_dir = f"TFA/coco/mining_iter{iter_}/{shot}shot_seed{seed}/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_dummy_iou.json"
                elif prefix == "DeFRCN":
                    json_dir = f"DeFRCN/coco/mining_iter{iter_}/{shot}shot_seed{seed}/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_dummy_iou.json"
                elif prefix == "imted":
                    json_dir = f"imted/coco/mining_iter{iter_}/imTED_{shot}shot_seed{seed}/imTED_{shot}shot_seed{seed}_trainval.bbox_filtrado_base2_nms_filt_score0.2_LC0.5_dummy_iou.json"
                else:
                    raise Exception(f"{prefix} not supported.") 

                _PREDEFINED_SPLITS_COCO["coco"][name] = ("coco/trainval2014", json_dir)


# VOC PSEUDOS FROM ITER1-4
# NOTE: CHANGE IF YOU ARE USING DIFFERENT PATHS!!!!
for prefix in ["TFA", "DeFRCN", "VitDet", "imted"]:
    for iter_ in [1,2,3,4]:
        for split in [1,2,3]:
            for shot in [1,2,3,5,10]:
                for seed in [0,1,2,3,4]:
                    name = "{}_voc_pseudos_{}shot_seed{}_split{}_iter{}".format(prefix, shot, seed, split, iter_)
                    if prefix == "TFA":
                        json_dir = f"TFA/voc/mining_iter{iter_}/split_{split}/{shot}shot_seed{seed}/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_dummy_iou.json"
                    elif prefix == "DeFRCN":
                        json_dir = f"DeFRCN/voc/mining_iter{iter_}/split_{split}/{shot}shot_seed{seed}/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_dummy_iou.json"
                    elif prefix == "VitDet":
                        json_dir = f"VitDet/voc/mining_iter{iter_}/split_{split}/{shot}shot_seed{seed}/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_dummy_iou.json"
                    elif prefix == "imted":
                        json_dir = f"imted/voc/mining_iter{iter_}/split_{split}/imTED_{shot}shot_seed{seed}/imTED_{shot}shot_seed{seed}_trainval.bbox_filtrado_base2_nms_filt_score0.2_LC0.5_dummy_iou.json"
                    else:
                        raise Exception(f"{prefix} not supported.") 

                    _PREDEFINED_SPLITS_VOC["voc"][name] = ("voc_coco/images", json_dir, int(split), "base_novel")


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_voc(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VOC.items():
        for key, (image_root, json_file, split, partition) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name, split, partition),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )