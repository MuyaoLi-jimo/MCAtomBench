#!/bin/bash

base_url=http://localhost:9020/v1
workers=10
max_frames=400
temperature=0.6
history_num=2
action_chunk_len=1
instruction_type="normal"
model_name_or_path="/public/models/JarvisVLA-qwen2-vl-7b"
model_local_path="JarvisVLA-qwen2-vl-7b" #"mc-vla-qwen2-vl-7b-250311-A800-c32-e1-b16-a1"

tasks=(
kill/creeper-diamond_sword
kill/evoker-diamond_axe
kill/ocelot-diamond_sword
kill/silverfish-diamond_axe
kill/salmon-diamond_sword
kill/silverfish-diamond_sword
kill/strider-diamond_sword
kill/elder_guardian-diamond_sword
kill/zombie-diamond_axe
kill/ender_dragon-diamond_sword
kill/rabbit-diamond_axe
kill/cod-diamond_axe
kill/pig-diamond_sword
kill/piglin-diamond_sword
kill/zombie_villager-diamond_sword
kill/endermite-diamond_axe
kill/hoglin-diamond_axe
kill/drowned-diamond_sword
kill/cat-diamond_sword
kill/enderman-diamond_axe
kill/slime-diamond_axe
kill/evoker-diamond_sword
kill/ender_dragon-diamond_axe
kill/squid-diamond_sword
kill/witch-diamond_axe
kill/villager-diamond_axe
kill/mooshroom-diamond_axe
kill/piglin-diamond_axe
kill/pillager-diamond_axe
kill/stray-diamond_sword
kill/ghast-diamond_sword
kill/bee-diamond_sword
kill/rabbit-diamond_sword
kill/hoglin-diamond_sword
kill/strider-diamond_axe
kill/blaze-diamond_sword
kill/cow-diamond_sword
kill/tropical_fish-diamond_axe
kill/iron_golem-diamond_sword
kill/pufferfish-diamond_sword
kill/ocelot-diamond_axe
kill/chicken-diamond_sword
kill/wolf-diamond_sword
kill/endermite-diamond_sword
kill/llama-diamond_sword
kill/witch-diamond_sword
kill/wither_skeleton-diamond_axe
kill/wither-diamond_axe
kill/stray-diamond_axe
kill/vex-diamond_axe
kill/enderman-diamond_sword
kill/guardian-diamond_sword
kill/polar_bear-diamond_sword
kill/cave_spider-diamond_sword
kill/shulker-diamond_sword
kill/chicken-diamond_axe
kill/sheep-diamond_sword
kill/donkey-diamond_sword
kill/magma_cube-diamond_sword
kill/donkey-diamond_axe
kill/pillager-diamond_sword
kill/ghast-diamond_axe
kill/guardian-diamond_axe
kill/cod-diamond_sword
kill/zombie-diamond_sword
kill/zombified_piglin-diamond_axe
kill/cow-diamond_axe
kill/wither_skeleton-diamond_sword

)

for task in "${tasks[@]}"; do
    env_config="$task"

    # Evaluate
    num_iterations=3
    for ((i = 0; i < num_iterations; i++)); do
        python mcabench/evaluate/evaluate.py \
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
            --record True \
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