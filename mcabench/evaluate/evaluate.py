'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-23 23:15:37
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-04-03 20:16:48
'''
"""
任务清单：
3. 修改craft agent的逻辑
3. 跑coa
"""

import argparse
from rich import print,console
from pathlib import Path
import os
import hydra
from omegaconf import OmegaConf
import ray
import time

from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from mcabench.minestudio_plus.simulator.callbacks import (
    RecordCallback,
    FastResetCallback2,
    InitInventoryCallback,
    SummonMobsCallback,
    CommandsCallback,
    TeleportCallback,
)

from minestudio.simulator.callbacks import (
    SpeedTestCallback, 
    RewardsCallback, 
    TaskCallback, 
)
from mcabench.minestudio_plus.models import CraftWorker,SmeltWorker
from mcabench.evaluate import draw_utils
from mcabench.utils import file_utils
from mcabench.agents import agent_wrapper

CFG_DIR = Path(__file__).parents[2]/"data"/"task_config"
        

def evaluate(video_path,evaluate_config:dict, agent_config:dict):

    # 打开yaml config
    env_cfg_path = CFG_DIR /  f"{evaluate_config['env_config']}.yaml"
    base_cfg_path = env_cfg_path.parent / "base.yaml"
    base_cfg = OmegaConf.load( base_cfg_path )
    env_cfg = OmegaConf.load( env_cfg_path)
    env_cfg = OmegaConf.merge(base_cfg, env_cfg)
    
    # 写入callback
    camera_cfg = CameraConfig(**env_cfg.camera_config)
    record_callback = RecordCallback(record_path=Path(video_path).parent, fps=evaluate_config["fps"], 
                                      show_actions= "action" in evaluate_config["demo"],show_instruction="instruction" in evaluate_config["demo"],
                                      record_actions=evaluate_config["record"],record_infos=evaluate_config["record"],record_raw_observation=(not evaluate_config["demo"] or evaluate_config["record"]))  
    callbacks = [
        FastResetCallback2(
            biomes=env_cfg.candidate_preferred_spawn_biome,
            random_tp_range=env_cfg.random_tp_range,
            start_time=env_cfg.start_time,
        ), 
        SpeedTestCallback(50), 
        TaskCallback(getattr(env_cfg,"task_conf",None)),
        RewardsCallback(getattr(env_cfg,"reward_conf",None)),
        InitInventoryCallback(env_cfg.init_inventory,
                                inventory_distraction_level=env_cfg.inventory_distraction_level,
                                equip_distraction_level=getattr(env_cfg,"equip_distraction_level","normal")
                                ),
        CommandsCallback(getattr(env_cfg,"command",[]),),
        record_callback,
    ]
    if hasattr(env_cfg,"teleport"):
        callbacks.append(TeleportCallback(x=env_cfg.teleport.x, y=env_cfg.teleport.y, z=env_cfg.teleport.z,))
    if env_cfg.mobs:
        callbacks.append(SummonMobsCallback(env_cfg.mobs))
    
    # init env
    env =  MinecraftSim(
        action_type="env",
        seed=env_cfg.seed,
        obs_size=env_cfg.origin_resolution,
        render_size=env_cfg.resize_resolution,
        camera_config=camera_cfg,
        preferred_spawn_biome=getattr(env_cfg,"preferred_spawn_biome",None),
        callbacks = callbacks
    )
    obs, info = env.reset()

    # 把环境准备好
    pre_agent = None
    worker_type =  getattr(env_cfg,"worker", None)
    if worker_type == "craft":
        pre_agent = CraftWorker(env,if_discrete=True)
    elif worker_type == "smelt":
        pre_agent = SmeltWorker(env,if_discrete=True)
    
    need_crafting_table = False
    if getattr(env_cfg, "need_gui", False):
        need_crafting_table= getattr(env_cfg,"need_crafting_table", False)
        need_furnace = getattr(env_cfg,"need_furnace", False)
        if need_crafting_table:
            try:
                frames,_,_ = pre_agent.open_crating_table_wo_recipe()
            except AssertionError as e:
                env.close()
                console.Console().log(f"error: {e}")
                return False,-1
        elif need_furnace:
            try:
                frames,_,_ = pre_agent.open_furnace_wo_recipe()
            except AssertionError as e:
                env.close()
                console.Console().log(f"error: {e}")
                return False,-1
        else:
            pre_agent._null_action(1)
            if not pre_agent.info['isGuiOpen']:
                pre_agent._call_func('inventory')
                
    record_callback.forget()

    agent = agent_wrapper.make_agent(**agent_config)
    env.action_type = agent.action_type
    agent.reset(env=env)

    success = (False,evaluate_config["max_frames"])
    start_time = time.time()
    for i in range(evaluate_config["max_frames"]):
        instructions = agent.get_instructions(env,env_cfg)
        observations = agent.get_observations(env,info)
        action = agent.forward(observations=observations,instructions=instructions,verbos=evaluate_config["verbos"])
        agent.show(record_callback)
        if evaluate_config["verbos"]:
            console.Console().log(action)
        obs, reward, terminated, truncated, info = env.step(action)

        if reward>0:
            success = (True,i)
            break   
        
    print(f"FPS: {success[1]/(time.time()-start_time)}")
    # sample another 30 steps if success
    if success[0]:
        for i in range(20):
            action = agent.forward([info["pov"]],instructions,verbos=evaluate_config["verbos"])
            agent.show(record_callback)
            obs, reward, terminated, truncated, info = env.step(action)
    # 最后一帧
    agent.show(record_callback)
    env.close()
    return success

@ray.remote
def evaluate_wrapper(video_path,evaluate_config,agent_config):
    success = evaluate(video_path=video_path,evaluate_config=evaluate_config,agent_config=agent_config)
    member_id = video_path.split("/")[-1].split(".")[0]
    return success[0],success[1],member_id

def multi_evaluate(args,agent_config,evaluate_config):
    
    model_ref_name = args.model_path.split('/')[-1]
    if "checkpoint" in model_ref_name:
        checkpoint_num = model_ref_name.split("-")[-1]
        model_base_name = args.model_path.split('/')[-2]
        model_ref_name = f"{model_base_name}-{checkpoint_num}"
    
    video_fold  = os.path.join(args.video_main_fold, f"{model_ref_name}-{args.env_config.split('/')[-1]}") 
    if not os.path.exists(video_fold):
        Path(video_fold).mkdir(parents=True,exist_ok=True)
    
    video_log_path = os.path.join(video_fold,"end.json") 
    resultss = file_utils.load_json_file(video_log_path,data_type="list")

    total_ids = [i for i in range(args.workers)]
    done_ids = [results[2] for results in resultss]
    undone_ids = [id for id in total_ids if str(id) not in done_ids]

    if not undone_ids:
        return
    
    ray.init()
    roll = len(undone_ids) // args.split_number + (1 if len(undone_ids) % args.split_number != 0 else 0)
    for i in range(roll):
        part_undone_ids = undone_ids[i*args.split_number:min((i+1)*args.split_number, len(undone_ids))]
        result_ids = [evaluate_wrapper.remote(video_path=os.path.join(video_fold,str(i),f"{i}.mp4"),evaluate_config=evaluate_config,agent_config=agent_config) for i in part_undone_ids]
        futures = result_ids
        
        while len(futures) > 0:
            ready_futures, rest_futures = ray.wait(futures,timeout=24*60*60)
            results = ray.get(ready_futures,timeout=60*60)  # Retrieve all results
            resultss.extend(results)
            # 写入日志文件
            file_utils.dump_json_file(resultss,video_log_path,if_backup=False)
            print(f"part frames IDs: {results} done!")
            futures = rest_futures
        
        ray.shutdown()
        
    draw_utils.show_success_rate(resultss,os.path.join(video_fold,"image.png") )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1) 
    parser.add_argument('--split-number', type=int, default=5) 
    parser.add_argument('--env-config',"-e", type=str,) 
    parser.add_argument('--agent-mode', type=str, default="rt2")
    parser.add_argument('--system-prompt-mode', type=str,default="")
    parser.add_argument('--video-main-fold',type=str)
    parser.add_argument('--max-frames', type=int, default=200) 
    parser.add_argument('--verbos', type=bool, default=False)
    parser.add_argument('--demo', type=str, default="") 
    parser.add_argument('--record', type=bool, default=False)
    parser.add_argument('--fps',type=int)
    
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--LLM_backbone', type=str,default="")
    parser.add_argument('--VLM_backbone', type=str,default="")
    parser.add_argument('--tokenizer_path', type=str,default="")

    parser.add_argument('--base-url',type=str,default="")
    parser.add_argument('--instruction-type',type=str, default='normal')
    parser.add_argument('--temperature','-t',type=float, default=0.7)
    parser.add_argument('--history-num',type=int, default=0)
    parser.add_argument('--action-chunk-len',type=int, default=1)

    args = parser.parse_args()


    if not args.base_url:
        args.base_url=None
        
    agent_config = dict(
        agent_mode = args.agent_mode,
        system_prompt_mode = args.system_prompt_mode,
        temperature=args.temperature,
        history_num = args.history_num,
        instruction_type = args.instruction_type,
        action_chunk_len = args.action_chunk_len,
        base_url=args.base_url,
        model_path=args.model_path,
        LLM_backbone = args.LLM_backbone,
        VLM_backbone = args.VLM_backbone,
        tokenizer_path = args.tokenizer_path,
        
    )
    evaluate_config = dict(
        env_config = args.env_config,
        max_frames = args.max_frames,
        verbos = args.verbos,
        demo=args.demo,
        record = args.record,
        fps=args.fps
    )
    
    if args.workers==0:
        evaluate_config["verbos"] = True
        video_path = f"{args.model_path.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4"
        evaluate(video_path=video_path,evaluate_config = evaluate_config, agent_config=agent_config)
    elif args.workers==1:
        video_path = f"{args.model_path.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4"
        evaluate(video_path=video_path,evaluate_config = evaluate_config,agent_config=agent_config)
    elif args.workers>1:
        multi_evaluate(args,agent_config=agent_config,evaluate_config=evaluate_config)
        
    
