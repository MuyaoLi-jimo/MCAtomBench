import time

from rich import print
from openai import OpenAI
import random
from typing import Literal
import copy
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from minestudio.simulator.entry import MinecraftSim

from mcabench.agents.vla import load_model
from mcabench.agents import action_mapping, base_agent,vlm_client
from mcabench.utils.file_utils import load_json_file

#################
# prompt
#################

BASE_INSTRUCTION_TEMPLATE = [
    "help me to craft a {}.",
    "craft a {}.",
    "Craft a {}.",
    "Could you craft a {} for me?",
    "I need you to craft a {}.",
    "Please craft a {} in the game.",
    "Craft me a {} quickly.",
    "Make sure to craft a {} for the task.",
    "Craft a {} so I can use it.",
    "Let’s craft a {} for this project.",
    "I need you to craft {} right now.",
]

class RT2AGENT(vlm_client.VlMClient,base_agent.Agent):
    def __init__(self, model_path, base_url, system_prompt_mode, api_key="EMPTY",
                 LLM_backbone = "", VLM_backbone="",tokenizer_path="",
                 history_num=0,action_chunk_len=1, bpe=0,
                 instruction_type:Literal['simple','recipe','normal'] = 'normal',
                 temperature=0.5,max_tokens=1024,
                 **kwargs):
        
        base_agent.Agent.__init__(self, agent_mode="rt2",**kwargs)
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            model_path=model_path,
            system_prompt_mode=system_prompt_mode,
            agent_mode="rt2",
            **kwargs,
        )
        
        self._action_type = "agent"
        
        if not LLM_backbone:
            self.LLM_backbone,self.VLM_backbone = load_model.load_visual_model(checkpoint_path=model_path)
            tokenizer_path = model_path
        else:
            self.LLM_backbone = LLM_backbone
            self.VLM_backbone = VLM_backbone
        
        self.model_path = model_path
        
            
        if self.LLM_backbone in {"llama-3","llama-2","qwen2_vl"}:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,  
                trust_remote_code=True,
            )
            
        self.action_tokenizer = action_mapping.OneActionTokenizer(tokenizer_type=self.LLM_backbone)
        
        self.prompt_library = load_json_file(Path(__file__).parents[3]/"data"/"assets"/"instructions.json") #存储我写好的instructions
        self.recipe_fold=Path(__file__).parents[3]/"data"/"assets"/"recipes" # 存储所有recipes的文件夹
        self.recipes = dict()  #制作方案集合        
       
        self.actions = []
        self.action_chunk_len=action_chunk_len  # 一次返回一个action chunk
        # 用于带有记忆的agent
        self.history_num = history_num
        self.history = []
        
        self.instruction_type = instruction_type
            
        self.set_processor_wrapper(model_name=self.VLM_backbone)

    def reset(self,env:MinecraftSim):
        self.history = []
          
    def rule_based_instruction(self,env_prompt:str):
        item_name = env_prompt[11:].replace("_"," ")
        instruction_template = random.choice(BASE_INSTRUCTION_TEMPLATE)
        return instruction_template.format(item_name)
          
    def create_basic_instruction(self,env_prompt:str):
        instruction = "."
        instructions = self.prompt_library.get(env_prompt,{}).get("instruct")
        if instructions:
            instruction = np.random.choice(instructions)
        else:
            instruction = self.rule_based_instruction(env_prompt)

        if instruction.strip()[-1] != '.':
            instruction += ". \n"
        instruction += "\n"
        return instruction
        
    def get_recipe_item_name(self,ingredient:dict):
        item_name = ingredient.get("item") 
        if not item_name:
            item_name = ingredient.get("tag") 
        return item_name
        
    def create_recipe_prompt_from_library(self,item_name:str):
        if item_name in self.recipes:
            return self.recipes[item_name]
        recipe_path = self.recipe_fold/f"{item_name}.json"
        print(recipe_path)
        if not recipe_path.exists():
            self.recipes[item_name]= ""
            return ""
        recipe_file = load_json_file(recipe_path)
        recipe_type = recipe_file.get("type",None)
        
        prompt = ""
        if not recipe_type:
            self.recipes[item_name]= ""
            return ""
        elif recipe_type=="minecraft:crafting_shapeless":
            prompt+=f"\nYou will need the following ingredients: \n"
            ingredients = recipe_file.get("ingredients",None)
            ingredients_list = []
            for ingredient in ingredients:
                ingredient_name = self.get_recipe_item_name(ingredient)
                if not ingredient_name:
                    break
                ingredients_list.append(ingredient_name[10:].replace("_"," "))
            ingredients_dict = Counter(ingredients_list)
            for item,number in ingredients_dict.items():
                prompt += f"{number} {item}, "
            prompt += "\n"
        elif recipe_type == "minecraft:crafting_shaped":
            prompt+="\nArrange the materials in the crafting grid according to the following pattern: \n"
            patterns = recipe_file.get("pattern",None)
            if not patterns:
                return ""
            ingredients = recipe_file.get("key",{})
            ingredients_dict = {}
            for ingredient_mark,value in ingredients.items():
                ingredient_name = self.get_recipe_item_name(value)
                ingredients_dict[ingredient_mark] = ingredient_name[10:]
            prompt+="\n"
            for pattern_line in patterns:
                for pattern_mark in pattern_line:
                    ingredients_name = ""
                    if pattern_mark==" ":
                        ingredients_name = "air"
                    else:
                        ingredients_name = ingredients_dict.get(pattern_mark,"air")
                    prompt += f" {ingredients_name} |"
                if prompt[-1]=='|':
                    prompt = prompt[:-1] 
                    prompt += "\n"
            prompt +="\n"
        else:
            self.recipes[item_name]= ""
            return ""
        result_num = recipe_file.get("result",{}).get("count",1)
        prompt += f"and get {result_num} {item_name.replace('_',' ')}. \n"
        self.recipes[item_name]= prompt
        return prompt
        
    def create_recipe_prompt(self,env_prompt:str):
        """从原始的一句话转换成prompt """
        prompt = ""
        # else
        recipe = self.prompt_library.get(env_prompt,{}).get("recipe")
        if recipe:
            prompt += "\nArrange the materials in the crafting grid according to the following pattern: \n"
            prompt += recipe[0]
        else:
            item_name = env_prompt.replace(" ","_").split(":")[-1]
            prompt += self.create_recipe_prompt_from_library(item_name,)
        return prompt
        
    def create_detailed_instruction(self,env_prompt):
        prompt =None
        if self.instruction_type == 'recipe':
            prompt = self.create_basic_instruction(env_prompt)
            recipe_prompt = self.create_recipe_prompt(env_prompt)

            prompt += recipe_prompt
        elif self.instruction_type == 'simple':
            prompt = self.create_thought(env_prompt) 
        elif self.instruction_type == 'normal':
            natural_text = env_prompt.replace("_"," ").replace(":"," ")
            prompt = random.choice(self.prompt_library.get(env_prompt,{"instruct":[natural_text]})["instruct"])
        else:
            raise ValueError(f"do not set the instruction class {self.instruction_type}")
        return prompt
        
    def create_thought(self,env_prompt):
        thought = copy.copy(self.prompt_library.get(env_prompt,{}).get("thought"))
        if not thought:
            thought = env_prompt.replace("item",str(1)).replace("_"," ").replace(":"," ")   #craft item xxx =》 craft 1 item
        thought += '. \n'
        return thought
    
    def get_instructions(self,env,env_cfg):
        return [item["text"] for item in env_cfg.task_conf]
        
    def forward(self,observations:list,instructions:list,verbos=False):
        if self.actions:
            if verbos:
                print(self.actions)
            if len(self.actions)>1:
                return self.actions.pop(0)
            else:
                action = self.actions[0]
                self.actions = []
                return action
        messages = []
        if self.system_prompt:
            messages.append(self.processor_wrapper.create_system_prompt(system_prompt=self.system_prompt))
   
        image = self.processor_wrapper.create_image_input(observations[0]) 

        detailed_instruction = self.create_detailed_instruction(instructions[0])
        thought= self.create_thought(instructions[0]) if self.instruction_type =="recipe" else ""

        if self.history_num:
            if not self.history: #如果历史为空
                self.history = [(image,self.action_tokenizer.null_token(),copy.copy(thought),0)]*self.history_num
            new_history = [None]*self.history_num
            new_history[:-1] = self.history[1:]
            for hdx,(im, ac, past_thought,_) in enumerate(self.history):
                prompt_input = ""
                if self.instruction_type == 'recipe':
                    prompt_input = "\nthought: " + past_thought + "\nobservation: "  #往上一个prompt上加上这一步的thought
                else:
                    prompt_input = "\nobservation: "
                if not hdx: #hdx==0
                    prompt_input = detailed_instruction + prompt_input
                #print(ac,prompt_input,)
                messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[prompt_input],image=[im]))
                messages.append(self.processor_wrapper.create_message_vllm(role="assistant",input_type="text",prompt=[ac],))
            
        prompt_input = ""
        if self.instruction_type == 'recipe':
            prompt_input = "\nthought: " + thought + "\nobservation: "
        else:
            prompt_input = "\nobservation: "
        if not self.history_num:
            prompt_input = detailed_instruction + prompt_input

        messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[prompt_input],image=[image]))

        if_token_ids = True if self.use_vllm and self.LLM_backbone in {"qwen2_vl","llama-2","llama-3"} else False

        outputs,content = self.generate(messages=messages,verbos=verbos,if_token_ids=if_token_ids)
        
        if verbos:
            print(content)
        
        if self.history_num:
            new_history[-1] = (image,content,thought,self.history[-1][-1]+1)
            self.history = new_history
    
        actions =  self.action_tokenizer.decode(outputs)

        len_action = min(self.action_chunk_len,len(actions))
        self.actions = actions[:len_action]
        
        if verbos:
            print(actions)
        
        return self.actions.pop(0)
