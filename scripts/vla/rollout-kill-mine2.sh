#!/bin/bash

base_url=http://localhost:9021/v1
workers=10
max_frames=400
temperature=0.6
history_num=2
action_chunk_len=1
instruction_type="normal"
model_name_or_path="/public/models/JarvisVLA-qwen2-vl-7b"
model_local_path="JarvisVLA-qwen2-vl-7b" #"mc-vla-qwen2-vl-7b-250311-A800-c32-e1-b16-a1"

tasks=(
kill/cat-diamond_axe
kill/zombie_villager-diamond_axe
kill/dolphin-diamond_axe
kill/skeleton_horse-diamond_sword
kill/skeleton-diamond_axe
kill/bee-diamond_axe
kill/turtle-diamond_axe
kill/skeleton-diamond_sword
kill/sheep-diamond_axe
kill/cave_spider-diamond_axe
kill/vex-diamond_sword
kill/fox-diamond_sword
kill/zombified_piglin-diamond_sword
kill/husk-diamond_sword
kill/horse-diamond_sword
kill/mooshroom-diamond_sword
kill/parrot-diamond_axe
kill/tropical_fish-diamond_sword
kill/wolf-diamond_axe
kill/pufferfish-diamond_axe
kill/shulker-diamond_axe
kill/skeleton_horse-diamond_axe
kill/slime-diamond_sword
kill/iron_golem-diamond_axe
kill/spider-diamond_axe
kill/piglin_brute-diamond_sword
kill/blaze-diamond_axe
kill/drowned-diamond_axe
kill/turtle-diamond_sword
kill/ravager-diamond_sword
kill/pig-diamond_axe
kill/polar_bear-diamond_axe
kill/vindicator-diamond_sword
kill/piglin_brute-diamond_axe
kill/wither-diamond_sword
kill/parrot-diamond_sword
kill/husk-diamond_axe
kill/villager-diamond_sword
kill/creeper-diamond_axe
kill/elder_guardian-diamond_axe
kill/vindicator-diamond_axe
kill/bat-diamond_axe
kill/spider-diamond_sword
kill/ravager-diamond_axe
kill/phantom-diamond_sword
kill/phantom-diamond_axe
kill/salmon-diamond_axe
kill/horse-diamond_axe
kill/squid-diamond_axe
kill/fox-diamond_axe
kill/llama-diamond_axe
kill/bat-diamond_sword
kill/magma_cube-diamond_axe
kill/dolphin-diamond_sword
mine/netherrack
mine/stone_pressure_plate
mine/iron_ore
mine/purple_bed
mine/grass_block
mine/acacia_door
mine/stone_brick_stairs
mine/crafting_table
mine/blue_bed
mine/grass
mine/oak_slab
mine/dirt
mine/white_stained_glass_pane
mine/orange_tulip
mine/large_fern
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