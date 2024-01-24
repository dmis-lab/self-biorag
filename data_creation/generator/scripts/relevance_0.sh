#!/bin/bash

# batch_nums=(0 1 2 3)
batch_nums=(0)
# intervals=(0 4 8 12 16 20 24 28 32 36 40 44)
intervals=(44)

for interval in ${intervals[@]}; do
    for batch_num in ${batch_nums[@]}; do
        num=$(($batch_num + $interval))
        CUDA_VISIBLE_DEVICES=4 python run_reward_vllm.py --input_file $DATA_PATH/generator/batch/prompt_data_batch_$num.jsonl --model_name $MODEL/selfbiorag_7b_critic --task 'relevance' --inst_mode relevance_instruction --input_mode relevance_input --metric match --result_fp $DATA_PATH/generator/batch_output/relevance/selfrag_relevance_batch_$num.json --split test
    done
done