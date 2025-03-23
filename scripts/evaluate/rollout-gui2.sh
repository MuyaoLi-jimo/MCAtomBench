#!/bin/bash

base_url=http://localhost:9021/v1
workers=10
max_frames=400
temperature=0.9
history_num=2
action_chunk_len=1
instruction_type="recipe"
model_local_path="JarvisVLA-qwen2-vl-7b" #"mc-vla-qwen2-vl-7b-250311-A800-c32-e1-b16-a1"

tasks=(
    craft/clay
craft/sandstone_wall
craft/light_blue_bed
craft/acacia_button
craft/prismarine
craft/jungle_trapdoor
craft/acacia_fence_gate
craft/piston
craft/dark_oak_stairs
craft/cyan_carpet
craft/light_blue_banner
craft/nether_bricks
craft/packed_ice
craft/oak_button
craft/birch_pressure_plate
craft/oak_pressure_plate
craft/grindstone
craft/stone_bricks
craft/yellow_stained_glass
craft/stone_brick_wall
craft/dispenser
craft/pumpkin_seeds
craft/golden_boots
craft/andesite_stairs
craft/polished_andesite_slab
craft/dark_prismarine
craft/iron_leggings
craft/dark_oak_planks
craft/light_gray_stained_glass
craft/mossy_stone_bricks
craft/lapis_block
craft/lodestone
craft/light_gray_terracotta
craft/hopper
craft/lime_stained_glass_pane
craft/smooth_red_sandstone_stairs
craft/brown_concrete_powder
craft/melon
craft/light_gray_wool
craft/blue_banner
craft/purpur_stairs
craft/purpur_pillar
craft/nether_wart_block
craft/jungle_sign
craft/polished_blackstone_brick_slab
craft/orange_dye
craft/coal_block
craft/shield
craft/magenta_bed
craft/polished_granite_slab
craft/gray_concrete_powder
craft/cyan_concrete_powder
craft/heavy_weighted_pressure_plate
craft/iron_bars
craft/ender_eye
craft/sea_lantern
craft/cyan_wool
craft/iron_helmet
craft/birch_stairs
craft/oak_fence
craft/granite_wall
craft/dried_kelp_block
craft/yellow_dye
craft/iron_block
craft/spruce_sign
craft/magenta_carpet
craft/leather_helmet
craft/yellow_banner
craft/yellow_bed
craft/black_banner
craft/birch_slab
craft/blackstone_slab
craft/tnt_minecart
craft/blue_stained_glass
craft/polished_blackstone_wall
craft/note_block
craft/slime_block
craft/tripwire_hook
craft/spruce_slab
craft/prismarine_wall
craft/blue_wool
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