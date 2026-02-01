#!/bin/bash

set -x

model="/root/autodl-tmp/checkpoint/qwen2.5-math-7b-nait-stage1"

dataset="weepcat/MCRD_math-7b_7b"

tau=0.25

echo "Generating soft labels with model: $model, dataset: $dataset, tau: $tau"
python train/generate_soft_label.py \
    --model_name_or_path "$model" \
    --dataset "$dataset" \
    --tau "$tau" \
    --stage 1