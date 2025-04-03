#!/bin/bash

base_url=http://localhost:9101/v1
cuda_visible_devices=1,2,3,4
workers=15
max_frames=300
temperature=0.5
history_num=0
action_chunk_len=1
instruction_type="recipe"
model_name_or_path="/public/JARVIS/checkpoints2/mc-vla-qwen2-vl-7b-25040102-A800-c32-e1-b16-a1/checkpoint-400"
model_local_path="mc-vla-qwen2-vl-7b-25040102-A800-c32-e1-b16-a1" #"mc-vla-qwen2-vl-7b-250311-A800-c32-e1-b16-a1"

#/public/JARVIS/checkpoints2/mc-vla-llava-next-vicuna-250330-A800-c32-e1-b8-a1/checkpoint-2400

tasks=( #mc-vla-qwen2-vl-7b-250401-A800-c32-e1-b16-a1-800-crafting_table
    craft/crafting_table
)

for task in "${tasks[@]}"; do
    env_config="$task"

    # Evaluate
    num_iterations=3
    for ((i = 0; i < num_iterations; i++)); do
        CUDA_VISIBLE_DEVICES=$cuda_visible_devices python mcabench/evaluate/evaluate.py \
            --agent-mode "rt2" \
            --workers $workers \
            --env-config $env_config \
            --max-frames $max_frames \
            --temperature $temperature \
            --model-path $model_name_or_path \
            --demo "" \
            --video-main-fold "/publicX/lmy/evaluate/$model_local_path" \
            --base-url "$base_url" \
            --history-num $history_num \
            --instruction-type $instruction_type \
            --action-chunk-len $action_chunk_len \
            #--record True \
            #--verbos True \
            


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