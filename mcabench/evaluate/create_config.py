"""
crafting & smelting
1. 读取可以作为craft和smelt的任务
2. 拿到它的recipe
3. 写清楚奖励
4. 得到最终的yaml文件，并写入
"""
from typing import Literal
from pathlib import Path
import yaml
from tqdm import tqdm
import random
import math
from mcabench.utils import file_utils
from mcabench.minestudio_plus.models import CraftWorker


#SLOT_SEQ = [10,12,14,16,18,20,24,26,28,30,32,34,2,4,6,8,9,11,13,15,17,19,21,23,25,27,29,31,33,35,1,3,5,7,]
BASE_DIR = Path(__file__).parents[2]/"data"/"task_config"
RECIPE_PATH = Path(__file__).parents[2]/"data"/"assets"/"recipes"
ITEMS_LIBRARY_PATH = Path(__file__).parents[2]/"data"/"assets"/"mc_constants.1.16.json"
TAG_LIBRARY_PATH = Path(__file__).parents[2]/"data"/"assets"/"tag_items.json"
SOURCE_LIBRARY_DIR = Path(__file__).parents[2]/"data"/"source"


def sample_recipe(test_type:str=Literal["craft","smelt"],recipes_path:Path=RECIPE_PATH,num:int=1000):
    recipes_files = []
    recipe_index_path = recipes_path/"_type_index.json"
    recipes_index = file_utils.load_json_file(recipe_index_path)
    if test_type == "craft":
        recipes_files.extend(recipes_index["minecraft:crafting_shaped"])
        recipes_files.extend(recipes_index["minecraft:crafting_shapeless"])
    elif test_type == "smelt":
        recipes_files.extend(recipes_index["minecraft:smelting"])
    sample_num = min(num,len(recipes_files))
    recipe_names = random.choices(recipes_files,k=sample_num)
    recipe_paths= [recipes_path / recipe_name for recipe_name in recipe_names]
    return recipe_paths

def create_inventory(materials:dict,items_library:dict,result_num:int=1):  
    init_inventory = [] 
    for item,quantity in materials.items():
        item_info = items_library.get(item,{})

        stackSize = item_info.get("stackSize",64)
        total_quantity = quantity * result_num
        slot_nums = math.ceil(total_quantity / stackSize)
        for _ in range(slot_nums):
            slot_quantity = min(stackSize, total_quantity)
            init_inventory.append({
                "slot":"random" ,
                "type": item,
                "quantity": f">={slot_quantity}",
            })
            total_quantity -= slot_quantity
    return init_inventory

def dump_yaml_file(config_content:dict,type_name:str,yaml_name:str,):
    yaml_path = BASE_DIR/ type_name/f"{yaml_name}.yaml"
    print(yaml_path,config_content)
    with open(str(yaml_path), "w", encoding="utf-8") as file:
        yaml.dump(config_content, file, allow_unicode=True, default_flow_style=False, indent=2)
    if yaml_path.exists():
        print(f"successfully create {yaml_path}")

def create_craft_content(recipe_path, dis=False,
                         items_library_path:Path = ITEMS_LIBRARY_PATH,tag_library_path:Path=TAG_LIBRARY_PATH,
                         ):
    type_name = "craft"
    recipe = file_utils.load_json_file(recipe_path)
    
    items_library_dict = file_utils.load_json_file(items_library_path)["items"]
    items_library = {item["type"]:item for item in items_library_dict}
    tag_library = file_utils.load_json_file(tag_library_path)
    
    result_item = recipe["result"]["item"][len("minecraft:"):]
    result_num = 1

    need_crafting_table = CraftWorker.crafting_type(recipe)
    materials = {}
    
    def get_item_name(item_info:dict)->str:
        if item_info.get('item'):
            item = item_info.get('item')[10:]
        else:
            tag_item = item_info.get('tag')
            item_list = tag_library[tag_item]
            item = random.choice(item_list)[10:]
        if not item:
            raise AssertionError(item_info,item_list,item)
        return item

    if recipe.get("type") == "minecraft:crafting_shaped":
        items = recipe.get('key')
        for k, v in items.items():
            signal = k       
            item = get_item_name(v)
            quantity = 0 
            pattern = recipe.get('pattern')
            for i in range(len(pattern)):
                for j in range(len(pattern[i])):
                    if pattern[i][j] == signal:
                        quantity += 1
            materials[item] = quantity
    elif recipe.get("type") == "minecraft:crafting_shapeless":
        items = recipe.get('ingredients')
        for v in items:   
            item = get_item_name(v) 
            quantity = materials.get(item,0)
            quantity +=1
            materials[item] = quantity
    
    # 构造初始仓库
    init_inventory = create_inventory(materials=materials,items_library=items_library,result_num=result_num)
            
    if need_crafting_table:
        init_inventory.append({
            "slot":0,
            "type":"crafting_table",
            "quantity":1,
        }
    )
        
    config_content = {
        "defaults":["base","_self_"],
        "task_conf":[{
            "name": result_item,
            "text": f"craft_item:{result_item}",
        }],
        "reward_conf":[{
            "event": "craft_item",
            "max_reward_times": 1,
            "reward": 1.0,
            "objects": [result_item],
            "identity": f"craft {result_item}",
        }],
        "init_inventory":init_inventory,
        "need_crafting_table": need_crafting_table,
        "need_gui": True,
    }
    
    if dis:
        config_content["inventory_distraction_level"] = "easy"
    dump_yaml_file(config_content=config_content,type_name=type_name,yaml_name=result_item)
    
    return config_content

def create_smelt_content(recipe_path,dis=False,
                         tag_library_path:Path=TAG_LIBRARY_PATH,items_library_path = ITEMS_LIBRARY_PATH):
    
    type_name = "smelt"
    recipe = file_utils.load_json_file(recipe_path)
    
    items_library_dict = file_utils.load_json_file(items_library_path)["items"]
    items_library = {item["type"]:item for item in items_library_dict}
    tag_library = file_utils.load_json_file(tag_library_path)
    
    result_item = recipe["result"][10:]
    # 确定制造数量
    result_num = 1
    
    materials = {}

    def get_item_quantity(item_info:dict)->str:
        if item_info.get('item'):
            item = item_info.get('item')[10:]
        else:
            tag_item = item_info.get('tag')
            item_list = tag_library[tag_item]
            item = random.choice(item_list)[10:]
        return item
    
    materials[get_item_quantity(recipe.get('ingredient'))] = result_num
    
    # 构造初始仓库
    init_inventory = create_inventory(materials=materials,items_library=items_library,result_num=result_num)

    init_inventory.append({
        "slot":0,
        "type":"furnace",
        "quantity":1,
        }
    )
    init_inventory.append({
        "slot":"random",
        "type":"coal",
        "quantity":">3",
        }
    )
    
    config_content = {
        "defaults":["base","_self_"],
        "task_conf":[{
            "name": result_item,
            "text": f"craft_item:{result_item}",
        }],
        "reward_conf":[{
            "event": "craft_item",
            "max_reward_times": 1,
            "reward": 1.0,
            "objects": [result_item],
            "identity": f"craft {result_item}",
        }],
        "init_inventory":init_inventory,
        "need_furnace": True,
        "need_gui": True,
    }
    
    if dis:
        config_content["inventory_distraction_level"] = "easy"
    dump_yaml_file(config_content=config_content,type_name=type_name,yaml_name=type_name + result_item)
    return config_content


def create_mine_content(task_name:str,mine_entity):
    init_inventory = []
    tools = mine_entity.get("tool")
    if tools:
        init_inventory.append({
            "slot":"0",
            "type":random.choice(tools),
            "quantity":"1",
            }
        )
    seed = random.choice(mine_entity["seeds"])
    config_content = {
        "defaults":["base","_self_"],
        "seed": seed["seed"],
        "teleport":{
            "x": seed["position"][0],
            "y": seed["position"][1],
            "z": seed["position"][2],
        },
        "task_conf":[{
            "name": "mine",
            "text": task_name,
        }],
        "reward_conf":[{
            "event": "mine_block",
            "max_reward_times": 1,
            "reward": 1.0,
            "objects": [task_name.split(":")[-1]],
            "identity": f"{task_name}",
        }],
        "init_inventory":init_inventory,
        "inventory_distraction_level" : "normal"
    }
    
    type_name = "mine"
    dump_yaml_file(config_content=config_content,type_name=type_name,yaml_name=type_name+"-"+task_name.split(":")[-1])


def create_kill_content(task_name:str,kill_entity:dict):
    
    tools = kill_entity.get("tool")
    assert isinstance(tools,list)
    for tool in tools:
        init_inventory = []
        if tool:
            init_inventory.append({
                "slot":"0",
                "type":tool,
                "quantity":"1",
                }
            )
        seed = random.choice(kill_entity["seeds"])
        config_content = {
            "defaults":["base","_self_"],
            "seed": seed["seed"],
            "teleport":{
                "x": seed["position"][0],
                "y": seed["position"][1],
                "z": seed["position"][2],
            },
            "command": kill_entity["commands"],
            "mobs":[{
                "name": task_name.split(":")[-1],
                "number": 1,
                "range_x":  [-1, 1],
                "range_z":  [2, 8],
            }],
            "task_conf":[{
                "name": "kill",
                "text": task_name,
            }],
            "reward_conf":[{
                "event": "kill_entity",
                "max_reward_times": 1,
                "reward": 1.0,
                "objects": [task_name.split(":")[-1]],
                "identity": f"{task_name}",
            }],
            "init_inventory":init_inventory,
            "inventory_distraction_level" : "normal"
        }
        
        type_name = "kill"
        dump_yaml_file(config_content=config_content,type_name=type_name,yaml_name=task_name.split(":")[-1]+"-"+tool)


def create_config(task_name:str="",test_type:Literal["craft","smelt","mine","kill"]="craft"):

    if test_type == "craft":
        if task_name:
            recipe_path = RECIPE_PATH / f"{task_name}.json"
            create_craft_content(recipe_path=recipe_path)
        else:
            recipe_paths = sample_recipe(test_type="craft")
            for recipe_path in tqdm(recipe_paths):
                create_craft_content(recipe_path=recipe_path,dis=False)
    elif test_type == "smelt":
        if task_name:
            recipe_path = RECIPE_PATH / f"{task_name}.json"
            create_smelt_content(recipe_path=recipe_path)
        else:
            recipe_paths = sample_recipe(test_type="smelt")
            for recipe_path in tqdm(recipe_paths):
                create_smelt_content(recipe_path=recipe_path,dis=True)
    elif test_type == "mine":
        kill_benchmark = file_utils.load_json_file(SOURCE_LIBRARY_DIR/f"{test_type}.json")
        if task_name:
            mine_entity = kill_benchmark.get(task_name,{})
            create_mine_content(task_name=task_name,mine_entity=mine_entity)
        else:
            for new_task_name,mine_entity in tqdm(kill_benchmark.items()):
                create_mine_content(task_name=new_task_name,mine_entity=mine_entity)
    elif test_type == "kill":
        kill_benchmark = file_utils.load_json_file(SOURCE_LIBRARY_DIR/f"{test_type}.json")
        if task_name:
            kill_entity = kill_benchmark.get(task_name,{})
            create_kill_content(task_name=task_name,kill_entity=kill_entity)
        else:
            for new_task_name,kill_entity in tqdm(kill_benchmark.items()):
                create_kill_content(task_name=new_task_name,kill_entity=kill_entity)

    
    
def get_useful(test_types:list,separate_num:int=7):
    reference_paths = []
    for test_type in test_types:
        data_dir = BASE_DIR / test_type
        data_paths = list(data_dir.glob("*.yaml"))
        local_reference_paths = [ f"{test_type}/{data_path.stem}" for data_path in data_paths]
        reference_paths.extend(local_reference_paths)
    total_length = len(reference_paths)
    split_size = total_length // separate_num
    splits = [
        reference_paths[i * split_size : (i + 1) * split_size]
        for i in range(separate_num )
    ]
    with open(BASE_DIR/"list.txt", "w", encoding="utf-8") as txt_file:
        for split in splits:
            for idx, reference in enumerate(split):
                if "base" not in reference:
                    txt_file.write(f"{reference}\n")
            txt_file.write(f"\n\n\n~~~~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~~~~~\n\n\n\n")
    
if __name__ == "__main__":
    
    #create_config(test_type="mine")
    get_useful(test_types=["craft","smelt"],separate_num=1)
