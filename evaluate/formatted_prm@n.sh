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
    for dataset in "${datasets[@]}"
    do  
        echo "正在转换 $filename ..."
        python evaluate/formatted_prm@n.py \
            --filename ./evaluate/$model/$(basename $dataset).json \
            --output "./evaluate/$model/$(basename ${dataset%.*})_prm@k_format.json" \
        > "./evaluate/$model/$(basename ${dataset%.*})_transform_prm@k.log"
    done
done