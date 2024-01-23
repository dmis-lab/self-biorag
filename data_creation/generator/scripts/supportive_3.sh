#!/bin/bash

# batch_nums=(0 1 2 3)
batch_nums=(3)
# intervals=(0 4 8 12 16 20 24 28 32 36 40 44)
intervals=(44)

for interval in ${intervals[@]}; do
    for batch_num in ${batch_nums[@]}; do
        num=$(($batch_num + $interval))
        CUDA_VISIBLE_DEVICES=3 python run_reward_vllm.py --input_file /scratch/x2696a10/self-rag/data/231226/selfrag/batch/prompt_data_batch_$num.jsonl --model_name /scratch/x2696a10/self-rag/critic_lm/selfrag-7b-231226-selfbiorag-medcpt-5k-lr2e-5-ep3/  --task groudness --inst_mode ground_multi_instruction --input_mode ground_multi_input --metric match --result_fp /scratch/x2696a10/self-rag/data/231226/selfrag/batch_output/selfrag_support_batch_$num.json --split test
    done
done