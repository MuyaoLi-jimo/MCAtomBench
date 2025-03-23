#!/bin/bash

base_url=http://localhost:9014/v1
workers=10
max_frames=400
temperature=0.9
history_num=2
action_chunk_len=1
instruction_type="recipe"
model_local_path="JarvisVLA-qwen2-vl-7b" #"mc-vla-qwen2-vl-7b-250311-A800-c32-e1-b16-a1"

tasks=(
    craft/soul_campfire
craft/crimson_hyphae
craft/iron_axe
craft/soul_lantern
craft/golden_axe
craft/warped_slab
craft/cyan_bed
craft/blue_ice
craft/detector_rail
craft/stone_button
craft/polished_blackstone_brick_stairs
craft/sandstone
craft/diamond_hoe
craft/stripped_crimson_hyphae
craft/cobblestone_wall
craft/ladder
craft/jungle_door
craft/birch_fence
craft/stripped_warped_hyphae
craft/purple_banner
craft/scaffolding
craft/birch_fence_gate
craft/warped_hyphae
craft/bricks
craft/red_sandstone_slab
craft/arrow
craft/light_blue_terracotta
craft/green_terracotta
craft/polished_granite_stairs
craft/polished_blackstone_brick_wall
craft/iron_sword
craft/cobblestone_stairs
craft/sugar
craft/chiseled_stone_bricks
craft/gray_carpet
craft/hay_block
craft/mossy_stone_brick_slab
craft/oak_planks
craft/coal
craft/golden_chestplate
craft/andesite
craft/orange_banner
craft/clock
craft/bread
craft/honey_block
craft/light_blue_carpet
craft/golden_sword
craft/mossy_stone_brick_wall
craft/magma_cream
craft/birch_button
craft/dark_oak_pressure_plate
craft/blaze_powder
craft/stripped_acacia_wood
craft/nether_brick_wall
craft/cut_sandstone
craft/sandstone_stairs
craft/armor_stand
craft/chiseled_nether_bricks
craft/repeater
craft/spruce_trapdoor
craft/warped_planks
craft/netherite_block
craft/barrel
craft/orange_carpet
craft/lime_stained_glass
craft/jungle_slab
craft/golden_pickaxe
craft/spectral_arrow
craft/quartz_slab
craft/light_gray_dye
craft/quartz_stairs
craft/brown_terracotta
craft/chiseled_quartz_block
craft/lime_carpet
craft/white_stained_glass
craft/crimson_planks
craft/purple_terracotta
craft/diamond_block
craft/iron_ingot
craft/diamond_axe
craft/fermented_spider_eye

)

for checkpoint in  107 ; do  #800 900 1000 1100 1300
    echo "Running for checkpoint $checkpoint..."

    model_name_or_path="/public/models/JarvisVLA-qwen2-vl-7b"
    log_path_name="$model_local_path-$checkpoint-$env_file"

    for task in "${tasks[@]}"; do
        env_config="$task"

        # Evaluate
        num_iterations=3
        for ((i = 0; i < num_iterations; i++)); do
            python jarvisvla/evaluate/evaluate.py \
                --workers $workers \
                --env-config $env_config \
                --max-frames $max_frames \
                --temperature $temperature \
                --checkpoints $model_name_or_path \
                --video-main-fold "/publicX/lmy/evaluate/$model_local_path" \
                --base-url "$base_url" \
                --history-num $history_num \
                --instruction-type $instruction_type \
                --action-chunk-len $action_chunk_len \
                #--verbos True \
            # 如果 Python 脚本执行成功，则退出循环
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
done 