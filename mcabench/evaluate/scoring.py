from typing import Literal
from pathlib import Path
import copy
import argparse
from mcabench.utils import file_utils

CONFIG_DIR = Path(__file__).parents[2]/"data"/"task_config"
SCORE_DIR = Path(__file__).parents[2]/"data"/"score"

def get_task(test_types:list,add_task_names:list=None,type_name:str="")->list:
    task_names = []
    if add_task_names:
        task_names.extend(add_task_names)
    for test_type in test_types:
        data_dir = CONFIG_DIR / test_type
        data_paths = list(data_dir.glob(f"*.yaml"))
        local_reference_paths = [ data_path.stem for data_path in data_paths if data_path.stem!="base"]
        task_names.extend(local_reference_paths)
    print(task_names)
    return task_names
        
def get_score(scores:list):
    
    win_num = 0
    for score in scores:
        if score[0]:
            win_num += 1 
    return win_num
        
def scoring(test_types:list,record_dir:Path,agent_name:str,add_task_names:list=None):
    unready_list = []
    record_dict = {}
    accumulate_score = 0
    task_names = get_task(test_types,add_task_names)
    record_dir = copy.copy(record_dir) / agent_name
    score_dir = SCORE_DIR / agent_name
    score_dir.mkdir(parents=True,exist_ok=True)
    for task_name in task_names:
        target_dirs = list(record_dir.glob(pattern=f"{agent_name}-{task_name}"))
        if not target_dirs:
            unready_list.append(task_name)
            continue
        target_dir = target_dirs[0]
        target_record_path = target_dir / "end.json"
        if not target_record_path.exists():
            unready_list.append(task_name)
            continue
        target_record = file_utils.load_json_file(target_record_path)
        win_num = get_score(target_record)
        record_dict[task_name] = {
                "win":win_num,
                "total":len(target_record)
            }
        accumulate_score += win_num/len(target_record)
    score_path = score_dir/"score.json"
    unready_path = score_dir/"unready.json"
    file_utils.dump_json_file(record_dict,score_path,if_backup=False)
    file_utils.dump_json_file(unready_list,unready_path,if_backup=False)
    print("average score: ",accumulate_score/len(record_dict))
    #print(task_names)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-dir',type=str, default="/publicX/lmy/evaluate")
    parser.add_argument('--agent-name',type=str, default="JarvisVLA-qwen2-vl-7b")
    args = parser.parse_args()
    
    scoring(test_types=["mine",],record_dir = Path(args.record_dir),agent_name=args.agent_name)
    