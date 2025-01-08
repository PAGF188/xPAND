# Base data
# python3 training_data_scripts/selection_train_test_examples.py \
#     --gt /datasets/dataset_fsod/trainvalno5k.json \
#     --dets /datasets/BC_COCO/coco_instances_results.json

# # Finetuning data
# for seed in 0 1 2 3 4 5 6 7 8 9
# do
#     for shot in 10 30
#     do
#         mkdir /datasets/BC_COCO/${shot}shot_seed${seed}/
#         python3 training_data_scripts/generate_sintetic_finetuning_data.py \
#             --gt /datasets/dataset_fsod/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_novel.json \
#             --output /datasets/BC_COCO/${shot}shot_seed${seed}/

#         python3 training_data_scripts/selection_train_test_examples.py \
#             --dets /datasets/BC_COCO/${shot}shot_seed${seed}/full_box_${shot}shot_all_trainval_novel_sint_boxes.json \
#             --finetuning

#     done
# done

# SSOD
for per in 1 5 10
do
    mkdir -p /datasets/BC_COCO_SSOD/1folder_per${per}/
    python3 training_data_scripts/generate_sintetic_finetuning_data.py \
        --gt /datasets/ssl_annotations/semi_supervised/semi_supervised/instances_train2017.1@${per}.json \
        --output /datasets/BC_COCO_SSOD/1folder_per${per}/
done