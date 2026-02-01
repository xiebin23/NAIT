set -x
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-mirror

read -r -d '' training_commands <<EOF
openrlhf.cli.train_prm \
   --save_path /root/autodl-tmp/checkpoint/qwen2.5-math-7b-nait-stage1 \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --pretrain Qwen/Qwen2.5-Math-7B \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 1e-6 \
   --dataset weepcat/MCRD_math-7b_7b \
   --input_key input \
   --label_key value \
   --attn_implementation flash_attention_2 \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group prm \
   --use_tensorboard Qwen2.5-Math-7B/MCRD_math-7b_7b \
   --placeholder_token [PRM] \
   --reward_tokens [POS] [NEG] \
   --eval_dataset weepcat/ProcessBench \
   --eval_splits gsm8k math olympiadbench omnimath \
   --wandb_run_name qwen2.5-math-7b-nait-stage1
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
