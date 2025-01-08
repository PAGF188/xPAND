#/bin/bash

P=$1 # Note: Change to your path. Under $P must be the datasets/ folder

# To check if detectron2 is properly installed.
# Must be 0.6 @/detectron2/detectron2. If not make sure that singularity is using the detectron2 version of the
# singularity container and not the one installed on local (which is used by default!!) 
#Output must be (if not check your singularity installation):

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
# GPU *anonymized*        *anonymized*
# Driver version          520.61.05
# CUDA_HOME               /usr/local/cuda
# TORCH_CUDA_ARCH_LIST    Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;Ampere
# Pillow                  9.2.0
# torchvision             0.11.1+cu113 @/usr/local/lib/python3.8/dist-packages/torchvision
# torchvision arch flags  3.5, 5.0, 6.0, 7.0, 7.5, 8.0, 8.6
# fvcore                  0.1.5
# iopath                  0.1.9
# cv2                     4.2.0


python3 -m detectron2.utils.collect_env  

#1) Create voc_coco folder
mkdir ${P}datasets/voc_coco/
mkdir ${P}datasets/voc_coco/images

#2) Convert voc annotation files to coco format
python3 pascal2coco.py --path ${P}datasets/ --output ${P}datasets/voc_coco/

#3) Copy voc images.
cp ${P}datasets/VOC2007/JPEGImages/* ${P}datasets/voc_coco/images/
cp ${P}datasets/VOC2012/JPEGImages/* ${P}datasets/voc_coco/images/

#4) Unify 2007/2012 files. The resulting partition is the used in previous FSOD works
for split in 1 2 3
do
    python3 combine_json_voc.py --d1 ${P}datasets/voc_coco/voc_2007_trainval_all${split}.json --d2 ${P}datasets/voc_coco/voc_2012_trainval_all${split}.json --output ${P}datasets/voc_coco/voc_2007_2012_trainval_all${split}.json
    python3 combine_json_voc.py --d1 ${P}datasets/voc_coco/voc_2007_trainval_base${split}.json --d2 ${P}datasets/voc_coco/voc_2012_trainval_base${split}.json --output ${P}datasets/voc_coco/voc_2007_2012_trainval_base${split}.json

done
