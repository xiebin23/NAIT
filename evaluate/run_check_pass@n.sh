#!/bin/bash

set -e

models=(
    "qwen2.5-3b-instruct"
    "qwen2.5-7b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
)

datasets=(
    "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
    "weepcat/MATH-500"
    "weepcat/gsm8k"
)

for model in "${models[@]}"
do
    echo "Processing model: $model"
    for dataset in "${datasets[@]}"
    do  
        if [[ "$model" == *math* ]]; then
            batch_size=400
        else
            batch_size=100
        fi
        echo "正在校验 $dataset ..."
        python evaluate/run_check_pass@n.py \
            --filename ./evaluate/$model/$(basename $dataset).json \
            --model "deepseek-chat" \
            --batch_size $batch_size \
            --output "./evaluate/$model/$(basename ${dataset%.*})_pass@n_check.json" \
            --interval 10 \
        > "./evaluate/$model/$(basename ${dataset%.*})_pass@n_check.log"
    done
done