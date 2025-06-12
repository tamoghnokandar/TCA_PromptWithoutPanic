#!/bin/bash
if [ "$1" == "V" ]; then
    data_root='/15tb_scratch_data/ramya/datasets/imagenetv2'
elif [ "$1" == "A" ]; then
    data_root='/15tb_scratch_data/ramya/datasets/imagenet-adversarial'
else
    data_root='/15tb_scratch_data/ramya/datasets'
fi

testsets=$1
num_attributes=$2
#arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt_tca

alpha=10
beta=20



if [ "$num_attributes" -eq 1 ]; then
    echo "Using a single attribute"
    alpha=10
    beta=20
elif [ "$num_attributes" -eq 2 ]; then
    echo "Using two attributes"
    alpha=45
    beta=15
else
    echo "Invalid number of attributes: $num_attributes"
    exit 1
fi



if [ "$arch" == "ViT-B/16" ]; then
	output_file="Training_Runs/output_alpha_${alpha}_beta_${beta}_arch_VIT_num_attributes_${num_attributes}_dataset_${testsets}_ensemble.txt"
else
	output_file="Training_Runs/output_alpha_${alpha}_beta_${beta}_arch_RN_num_attributes_${num_attributes}_dataset_${testsets}_ensemble.txt"
fi


python ./tpt_classification_ensemble.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init} --run_type ${run_type} --I_augmix --alpha ${alpha} --beta ${beta} --num_attributes ${num_attributes} --multi_gpu > "$output_file" \
