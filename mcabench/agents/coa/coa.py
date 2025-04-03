import re
import numpy as np
from typing import List
from copy import deepcopy
from pathlib import Path
import textwrap
import cv2
from PIL import Image
from rich import print
import math
from mcabench.utils import file_utils
from mcabench.agents import base_agent,vlm_client
from mcabench.agents.coa import extract
from mcabench.agents.vla import action_mapping, load_model
from minestudio.simulator.entry import MinecraftSim
from minestudio.simulator.callbacks.callback import MinecraftCallback

MC_RESOLUTION = (640,360)

class CoaAgent(vlm_client.VlMClient,base_agent.Agent):
    def __init__(self, 
                 model_path, base_url, api_key="EMPTY",
                 temperature=0.5,max_tokens=1024,
                 **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        self._action_type = "env"
        self.model_path = model_path
        self.set_processor_wrapper()
    
        self.prompt_library = file_utils.load_json_file(Path(__file__).parents[3]/"data"/"assets"/"instructions.json") #存储我写好的instructions
        self.history = []
        
    def reset(self,env:MinecraftSim):
        self.history = []
        
    def get_instructions(self,env,env_cfg):
        return [item["text"] for item in env_cfg.task_conf]
    
    def rule_based_instruction(self,raw_instruction:str):
        return raw_instruction.replace("_"," ").replace(":","").replace("item","the")
    
    def create_restruct_instruction(self,raw_instruction:str):
        instruction = "."
        instructions = self.prompt_library.get(raw_instruction,{}).get("instruct")
        if instructions:
            instruction = np.random.choice(instructions)
        else:
            instruction = self.rule_based_instruction(raw_instruction)

        if instruction.strip()[-1] != '.':
            instruction += ". \n"
        instruction += "\n"
        return instruction

    def to_bgr_uint8(self,frame):
        """
        Convert frame to a 3-channel BGR, uint8, C-contiguous array for OpenCV.
        """
        if isinstance(frame, Image.Image):
            # Convert PIL to NumPy
            frame = np.array(frame)
            # Handle possible alpha channel or just convert RGB → BGR
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif not isinstance(frame, np.ndarray):
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        # Now frame is np.ndarray, ensure type is uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Ensure shape is [H, W, 3]
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {frame.shape}, expected [H, W, 3]")

        # Ensure it’s contiguous
        frame = np.ascontiguousarray(frame)

        return frame

    def show(self, record_callback:MinecraftCallback):
        frames = record_callback.frames
        
        thought = self.history[-1]["thought"]
        hierarchical_action = extract.extract_hierarchical_action(thought)
        recent_frame = frames[-1]
        recent_frame = self.to_bgr_uint8(recent_frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX # cv2.FONT_HERSHEY_PLAIN # cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)
        radius = 4
        circle_color = (255,255,255)
        action_color =  (255,255,0)
        thickness = 1
        line_spacing=5
        
        #print(hierarchical_action)
        #print(type(recent_frame))
        
        radius_thresh = 10

        # 先画 grounding
        for g_item in hierarchical_action["grounding"]:
            x_g, y_g = g_item["point"][0]
            caption_g = g_item["action"]
            cv2.circle(recent_frame, (x_g, y_g), radius, action_color, -1)
            cv2.putText(recent_frame, caption_g, (x_g + 15, y_g), 
                        font, font_scale, action_color, thickness)

        # 再画 point
        for p_item in hierarchical_action["point"]:
            x_p, y_p = p_item["point"][0]
            caption_p = p_item["label"]

            # 判断和 grounding 的距离是否小于阈值
            hide_text = False
            for g_item in hierarchical_action["grounding"]:
                x_g, y_g = g_item["point"][0]
                dist = math.sqrt((x_p - x_g)**2 + (y_p - y_g)**2)
                if dist < radius_thresh:
                    # 如果任意一个 grounding 点距离小于 10，就不显示文字
                    hide_text = True
                    break

            # 画圆
            cv2.circle(recent_frame, (x_p, y_p), radius, circle_color, -1)

            # 如果不需要隐藏文字，则显示文字
            if not hide_text:
                cv2.putText(recent_frame, caption_p, (x_p + 15, y_p), 
                            font, font_scale, circle_color, thickness)
        
        # show thought
        wrapped_text = textwrap.wrap(repr(thought), width=70)
        
        #  创建一个白底文本区域图像
        max_length = 12
        line_height = int(cv2.getTextSize("A", font, font_scale, thickness)[0][1] + line_spacing)//2*2
        text_block = np.ones((line_height*max_length, MC_RESOLUTION[0], 3), dtype=np.uint8) * 255
        wrapped_text = wrapped_text[:max_length]
        
        for i, line in enumerate(wrapped_text):
            y = (i + 1) * line_height - line_spacing
            cv2.putText(text_block, line, (10, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        frames[-1] = np.vstack((recent_frame, text_block))
        record_callback.frames = frames
        
    def forward(self, observations, instructions, verbos=False):
        return super().forward(observations, instructions, verbos)


class LatentCoaAgent(CoaAgent):
    def __init__(self, 
                 model_path, base_url, api_key="EMPTY",
                 temperature=0.5,max_tokens=1024,
                 LLM_backbone = "", VLM_backbone="",tokenizer_path="",
                 **kwargs):
        super().__init__(
            model_path=model_path,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            LLM_backbone=LLM_backbone,VLM_backbone=VLM_backbone,tokenizer_path=tokenizer_path,
            **kwargs
        )
        self._action_type = "agent"
        
        if not LLM_backbone:
            self.LLM_backbone,self.VLM_backbone = load_model.load_visual_model(checkpoint_path=model_path)
            tokenizer_path = model_path
        else:
            self.LLM_backbone = LLM_backbone
            self.VLM_backbone = VLM_backbone
                
        self.action_tokenizer = action_mapping.OneActionTokenizer(tokenizer_type=self.LLM_backbone)
    
    def forward(self,observations:list,instructions:list,verbos=False):
        messages = []
        image = self.processor_wrapper.create_image_input(observations[0]) 
        instruction = self.create_restruct_instruction(instructions[0])
        
        messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[instruction],image=[image]))
        outputs,content = self.generate(messages=messages,verbos=verbos)
        if verbos:
            print(content)
        
        actions =  self.action_tokenizer.decode(outputs)
        
        self.history = [{"action":actions[0],"thought":content}]
        return actions[0]

class RawActionCoaAgent(CoaAgent):
    def __init__(self, 
                 model_path, base_url, api_key="EMPTY",
                 temperature=0.5,max_tokens=1024,
                 **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        self._action_type = "env"
        self.no_op = None

    def reset(self,env:MinecraftSim):
        self.no_op = env.noop_action()
        self.history = []
        
    def action_parse(self,raw_input:str)->List:
        matches = re.findall(r"<raw>(.*?)</raw>", raw_input)
        actions = []
        if not matches:
            return [self.no_op.copy()]
        for match in matches:
            action = deepcopy(self.no_op)
            camera_matches = re.findall(r"\(([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)\)", match)
            if camera_matches:
                action["camera"] = np.array([float(camera_matches[0][0]), float(camera_matches[0][1])])
            
            action_keys = ['attack','forward','back','left','right','jump','inventory','use','sprint','sneak', 'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5', 'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9']
            for action_key in action_keys:
                if action_key in match:
                    action[action_key] = 1
            actions.append(action)
        return actions
        
    def forward(self,observations:list,instructions:list,verbos=False):
        messages = []
        image = self.processor_wrapper.create_image_input(observations[0]) 
        instruction = self.create_restruct_instruction(instructions[0])
        
        messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[instruction],image=[image]))
        outputs,content = self.generate(messages=messages,verbos=verbos)
        if verbos:
            print(content)
            
        actions = self.action_parse(outputs)
        if verbos:
            print(actions)
            
        self.history = [{"action":actions[0],"thought":outputs}]
        return actions[0]
        

