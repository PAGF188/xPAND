#!/bin/bash
GPUS=$1
CONFIG_FILE=$2
TRAIN_FILE=$3
TEST_FILE=$4
OUTPUT_DIR=$5
INIT_WEIGHTS=$6

nvidia-smi
python3 -m detectron2.utils.collect_env # To debug. Output must be something like:

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


cd datasets
ln -sf /datasets/coco ./
ln -sf /datasets/cocosplit ./
ln -sf /datasets/dataset_fsod ./
ln -sf /datasets/voc_coco ./
ln -sf /datasets/VOC2007 ./
ln -sf /datasets/VOC2012 ./
ln -sf /datasets/vocsplit ./

ln -sf /datasets/BC_COCO ./   # Make sure you build this directory with correct data!
ln -sf /datasets/BC_VOC ./    # Make sure you build this directory with correct data!
cd ..
ls datasets

if [ $INIT_WEIGHTS = "None" ]; then
    python3 /detectron2/tools/lazyconfig_train_net.py --num-gpus $GPUS --config-file $CONFIG_FILE \
        dataloader.train.dataset.names="${TRAIN_FILE}" \
        dataloader.test.dataset.names="${TEST_FILE}" \
        train.output_dir="${OUTPUT_DIR}" \
        dataloader.evaluator.output_dir="${OUTPUT_DIR}/inference"
else
    python3 /detectron2/tools/lazyconfig_train_net.py --num-gpus $GPUS --config-file $CONFIG_FILE \
        dataloader.train.dataset.names="${TRAIN_FILE}" \
        dataloader.test.dataset.names="${TEST_FILE}" \
        train.output_dir="${OUTPUT_DIR}" \
        dataloader.evaluator.output_dir="${OUTPUT_DIR}/inference" \
        train.init_checkpoint="${INIT_WEIGHTS}"
fi
