# Base data
for split in 1 2 3
do
python3 training_data_scripts/selection_train_test_examples.py \
    --gt /datasets/voc_coco/voc_2007_2012_trainval_base${split}.json \
    --dets /datasets/BC_VOC/base${split}/coco_instances_results.json
done


#Finetuning data
mkdir /datasets/BC_VOC/finetunings
for split in 1 2 3
do
    for seed in 0 1 2 3 4
    do
        for shot in 1 2 3 5 10
        do
            python3 training_data_scripts/generate_sintetic_finetuning_data.py \
                --gt /datasets/voc_coco/voc_2007_trainval_novel${split}_${shot}shot_seed${seed}.json \
                --output /datasets/BC_VOC/finetunings/

            python3 training_data_scripts/selection_train_test_examples.py \
                --dets /datasets/BC_VOC/finetunings/voc_2007_trainval_novel${split}_${shot}shot_seed${seed}_sint_boxes.json \
                --finetuning
        done
    done
done