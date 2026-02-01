#!/bin/bash

set -e

models=(
    "qwen2.5-3b-instruct"
    "qwen2.5-7b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
)

for model in "${models[@]}"
do
    for filename in ./evaluate/$model/*_pass@n_check.json
    do  
        echo "正在转换 $filename ..."
        python evaluate/formatted_pass@n.py \
            --filename $filename \
            --output "./evaluate/$model/$(basename ${filename%_check.*})_pass@k_format.json" \
        > "./evaluate/$model/$(basename ${filename%_check.*})_transform_pass@k.log"
    done
done