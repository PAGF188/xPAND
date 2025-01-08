#!/bin/bash
GPUS=$1
seed=$2
shot=$3
pseudo_path=$4
pseudo_initial_name=$5
method=$6
iter=$7
split=$8

nvidia-smi
python3 -m detectron2.utils.collect_env # To debug. Output must be something like:
# ----------------------  --------------------------------------------------------------------
# sys.platform            linux
# Python                  3.8.10 (default, Jun 22 2022, 20:18:18) [GCC 9.4.0]
# numpy                   1.23.3
# detectron2              0.6 @/detectron2/detectron2
# Compiler                GCC 9.4
# CUDA compiler           CUDA 11.3
# detectron2 arch flags   3.5, 3.7, 5.0, 5.2, 5.3, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6
# DETECTRON2_ENV_MODULE   <not set>
# PyTorch                 1.10.0+cu113 @/usr/local/lib/python3.8/dist-packages/torch
# PyTorch debug build     False
# GPU available           Yes
# GPU *anonymous*         *anonymous*
# Driver version          520.61.05
# CUDA_HOME               /usr/local/cuda
# TORCH_CUDA_ARCH_LIST    Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;Ampere
# Pillow                  9.2.0
# torchvision             0.11.1+cu113 @/usr/local/lib/python3.8/dist-packages/torchvision
# torchvision arch flags  3.5, 5.0, 6.0, 7.0, 7.5, 8.0, 8.6
# fvcore                  0.1.5
# iopath                  0.1.9
# cv2                     4.2.0
# ----------------------  --------------------------------------------------------------------


mkdir datasets
cd datasets
ln -sf /datasets/coco ./
ln -sf /datasets/cocosplit ./
ln -sf /datasets/dataset_fsod ./
ln -sf /datasets/voc_coco ./
ln -sf /datasets/VOC2007 ./
ln -sf /datasets/VOC2012 ./
ln -sf /datasets/vocsplit ./

ln -sf /models/TFA ./
ln -sf /models/DeFRCN ./
ln -sf /models/VitDet ./
ln -sf /models/imted ./
cd ..
ls datasets


python3 pipeline_scripts/1_auxiliary_add_dumm_iou_BV.py --gt /datasets/voc_coco/voc_2007_2012_trainval_all${split}.json \
    --dets ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC0.5.json

TEST_NAME=${method}_voc_pseudos_${shot}shot_seed${seed}_split${split}_iter${iter}  # MAKE SURE YOU HAVE REGISTERED THE CC OUTPUT DATASETS IN builtin.py !!!
OUTPUT_DIR=${pseudo_path}/BC_output/
WEIGHTS=/models/BC_VOC/split${split}/${shot}shot_seed${seed}/model_final.pth
python3 /detectron2/tools/lazyconfig_train_net.py --eval-only --num-gpus $GPUS --config-file configs/box_ref_voc.py \
    dataloader.test.dataset.names="${TEST_NAME}" dataloader.evaluator.output_dir="${OUTPUT_DIR}" \
    train.init_checkpoint="${WEIGHTS}" train.output_dir="${OUTPUT_DIR}"

python3 pipeline_scripts/2_iou_filtering_BV.py --pseudos ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC0.5.json \
    --box_ver_output ${pseudo_path}/BC_output/${method}_voc_pseudos_${shot}shot_seed${seed}_split${split}_iter${iter}_ubbr.json \
    --th 0.8

python3 pipeline_scripts/3_filter_score_BV.py --dets ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json \
    --score_th 0.5

python3 pipeline_scripts/4_selection_random.py --dets ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8_score0.5.json \
    --factor 10

# For imted (which is the only tested detector that is not built on detectron2) the flag to ignore regions is different
if [ $method = "imted" ]; then
    python3 pipeline_scripts/5_add_ignore_regions_imted.py \
    --ps ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8_score0.5_random_selection.json \
    --ig ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2.json \
    --score 0.25
else
    python3 pipeline_scripts/5_add_ignore_regions.py \
    --ps ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8_score0.5_random_selection.json \
    --ig ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2.json \
    --score 0.25
fi

# # Additional step for voc (the combination in COCO is made, through the config file, when training the final detector)
python3 pipeline_scripts/6_json_union.py --ps ${pseudo_path}/${pseudo_initial_name}_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8_score0.5_random_selection_ignore_regions0.25_interseccion_nms.json \
        --gt /datasets/voc_coco/voc_2007_trainval_novel${split}_${shot}shot_seed${seed}.json