export HF_HOME=/root/autodl-tmp/hf-mirror
export HF_ENDPOINT="https://hf-mirror.com"


# trap 'echo "[$(date)] 脚本退出，10秒后关机... "; sleep 10; /bin/shutdown -h now' EXIT

datasets=(
    "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
    "weepcat/MATH-500"
    "weepcat/gsm8k"
)

prm_models=(
    "Qwen/Qwen2.5-Math-PRM-7B"
    "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"
    "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    "peiyi9979/math-shepherd-mistral-7b-prm"
    "PRIME-RL/EurusPRM-Stage1"
    "PRIME-RL/EurusPRM-Stage2"
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
    "/root/autodl-tmp/checkpoint/qwen2.5-math-7b-nait-stage1"
    "/root/autodl-tmp/checkpoint/qwen2.5-math-7b-nait-stage2"
    "/root/autodl-tmp/checkpoint/qwen2.5-math-7b-nait-stage3"
)


for prm_model in "${prm_models[@]}"
do  
    filename="./evaluate/$prm_model/$(basename ${dataset%.*}).log"
    echo "使用 PRM 模型: $prm_model"
    vllm serve \
        $prm_model \
        --port 8000 \
        --tensor-parallel-size 8 \
        --dtype auto \
        --enable-prefix-caching

    python evaluate/run_eval_prm_rlhf.py \
        --filename "$datasets" \
        --model_name "$prm_model" \
        --output "$filename" \
done