#!/bin/bash

base_url=http://localhost:9014/v1
workers=10
max_frames=300
temperature=0.7
history_num=0
action_chunk_len=1
instruction_type="recipe"
model_name_or_path="/data/JARVIS/checkpoints2/mc-coa-qwen2-vl-7b-25040303-A800-c32-e1-b32-a1"
model_local_path="mc-coa-qwen2-vl-7b-25040303-A800-c32-e1-b32-a1" #"mc-vla-qwen2-vl-7b-250311-A800-c32-e1-b16-a1"

tasks=(
    craft/diamond_sword
    craft/ladder
    craft/bread
    craft/golden_leggings
)

for task in "${tasks[@]}"; do
    env_config="$task"

    # Evaluate
    num_iterations=3
    for ((i = 0; i < num_iterations; i++)); do
        python mcabench/evaluate/evaluate.py \
            --agent-mode "latent-coa" \
            --system-prompt-mode "rt2" \
            --workers $workers \
            --env-config $env_config \
            --max-frames $max_frames \
            --temperature $temperature \
            --model-path $model_name_or_path \
            --demo "instruction" \
            --video-main-fold "/publicX/lmy/evaluate/$model_local_path" \
            --base-url "$base_url" \
            --history-num $history_num \
            --instruction-type $instruction_type \
            --action-chunk-len $action_chunk_len \
            --fps 30 \
            #--verbos True \
            #--record True \


        if [[ $? -eq 0 ]]; then
            echo "第 $i 次迭代中的 Python 脚本执行成功，退出循环。"
            break
        fi
        if [[ $i -lt $((num_iterations - 1)) ]]; then
            echo "等待 10 秒..."
            sleep 10
        fi
    done
done 