for split in 2
do
    for seed in 0 
    do
        for shot in 5
        do
            echo "seed: $seed, shot: $shot" 
            python3 stop_criterion_vfast.py \
                --dets1 /datasets/imted/mining_iter2/split_${split}/imTED_${shot}shot_seed${seed}/imTED_${shot}shot_seed${seed}_trainval.bbox_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json \
                --dets2 /datasets/imted/mining_iter3/split_${split}/imTED_${shot}shot_seed${seed}/imTED_${shot}shot_seed${seed}_trainval.bbox_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json  >> ./reuslts_imted_parar_iter2_3.txt
                
                #--dets1 /datasets/TFA/voc/faster_rcnn/mining_iter2/split_${split}/${shot}shot_seed${seed}/inference/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json \
                #--dets2 /datasets/TFA/voc/faster_rcnn/mining_iter3/split_${split}/${shot}shot_seed${seed}/inference/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json >> ./reuslts_tfa_parar_iter2_3.txt

                # --dets1 /datasets/VitDet/mining_iter2/split_${split}/${shot}shot_seed${seed}/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json \
                # --dets2 /datasets/VitDet/mining_iter3/split_${split}/${shot}shot_seed${seed}/inference/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json >> ./reuslts_vitdet_parar_iter2_3.txt
                #--dets1 /datasets/VitDet/mining/split_${split}/${shot}shot_seed${seed}/coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8.json \
        done
    done
done

#coco_instances_results_filtrado_base2_nms_filt_score0.2_LC0.5_box_verification0.8_score0.5_random_selection.json