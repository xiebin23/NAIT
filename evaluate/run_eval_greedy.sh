#!/bin/bash

set -e

models=(
    "qwen2.5-3b-instruct"
    "qwen2.5-7b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
)

datasets=(
    "weepcat/gsm8k"
    "weepcat/MATH-500"
    "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
)

for model in "${models[@]}"
do  
    echo "Processing model: $model"
    for dataset in "${datasets[@]}"
    do  
        echo "正在统计 $dataset ..."
        python evaluate/run_eval_greedy.py \
            --filename ./evaluate/$model/$(basename $dataset)_greedy_format.json \
            --output ./evaluate/$model/stats_greedy.txt \
        > "./evaluate/$model/$(basename ${dataset%.*})_greedy.log"
    done
done