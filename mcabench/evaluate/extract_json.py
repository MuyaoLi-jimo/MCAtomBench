from pathlib import Path
from omegaconf import OmegaConf
from mcabench.utils import file_utils
CFG_DIR = Path(__file__).parents[2]/"data"/"task_config"

def extract_craft():
    env_cfg_dir = CFG_DIR / "craft"
    data_paths = set(env_cfg_dir.glob("*.yaml"))
    data_paths.remove(env_cfg_dir/"base.yaml")
    config_datas = {}
    for env_cfg_path in data_paths:
        env_cfg = OmegaConf.load( env_cfg_path)
        need_crafting_table = bool(env_cfg.need_crafting_table)
        task_name = env_cfg.task_conf[0].text
        init_inventory = [OmegaConf.to_container(init_inventory) for init_inventory in env_cfg.init_inventory]
        if need_crafting_table:
            init_inventory = init_inventory[:-1]
        config_datas[task_name] = {
            "seeds": [{
                "seed": 1,
                "position": [
                    -1838,
                    72,
                    30501
                ]}
            ],
            "need_crafting_table": need_crafting_table,
            "init_inventory": init_inventory,
            "goal": env_cfg.reward_conf[0].objects[0],
        }
    file_utils.dump_json_file(config_datas,CFG_DIR/"craft.json")


def extract_smelt():
    env_cfg_dir = CFG_DIR / "smelt"
    data_paths = set(env_cfg_dir.glob("*.yaml"))
    data_paths.remove(env_cfg_dir/"base.yaml")
    config_datas = {}
    for env_cfg_path in data_paths:
        env_cfg = OmegaConf.load( env_cfg_path)
        need_furnace = bool(env_cfg.need_furnace)
        task_name = env_cfg.task_conf[0].text
        init_inventory = [OmegaConf.to_container(init_inventory) for init_inventory in env_cfg.init_inventory]
        if need_furnace:
            init_inventory = init_inventory[:-2] + [init_inventory[-1]]
        config_datas[task_name] = {
            "seeds": [{
                "seed": 1,
                "position": [
                    -1838,
                    72,
                    30501
                ]}
            ],
            "need_furnace": need_furnace,
            "init_inventory": init_inventory,
            "goal": env_cfg.reward_conf[0].objects[0],
        }
    file_utils.dump_json_file(config_datas,CFG_DIR/"smelt.json")

if __name__ == "__main__":
    extract_craft()
    extract_smelt()