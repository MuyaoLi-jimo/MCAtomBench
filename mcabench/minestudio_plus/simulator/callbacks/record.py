'''
Date: 2024-11-11 16:40:57
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-24 04:10:49
FilePath: /MineStudio/minestudio/simulator/callbacks/record.py
'''
import av
from pathlib import Path
from minestudio.simulator.callbacks.callback import MinecraftCallback
from typing import Literal
from rich import print
from copy import deepcopy
import numpy as np
from gymnasium import spaces
from collections import defaultdict
from omegaconf import DictConfig
import json
import cv2


class RecordCallback(MinecraftCallback):
    def __init__(self, record_path: str, fps: int = 20, frame_type: Literal['pov', 'obs'] = 'pov', recording: bool = True,
                    show_actions=False,show_instruction=False, 
                    record_actions=False,record_infos=False, record_raw_observation = True,
                    record_npy_observation=False, 
                 **kwargs):
        #print("record_actions ",record_actions,"record_infos ",record_infos,"record_raw_observation ",record_raw_observation,"show_instruction ",show_instruction,"show_actions ",show_actions)
        super().__init__(**kwargs)
        self.record_path = Path(record_path)
        self.record_path.mkdir(parents=True, exist_ok=True)
        self.recording = recording
        self.record_actions = record_actions
        self.show_actions = show_actions
        self.show_instruction = show_instruction
        self.record_raw_observation = record_raw_observation
        self.record_infos = record_infos
        self.record_origin_observation = record_npy_observation
        if recording:
            print(f'[green]Recording enabled, saving episodes to {self.record_path}[/green]')
        self.fps = fps
        self.frame_type = frame_type
        self.episode_id = 0
        self.frames = []
        self.infos = []
        self.actions = []
        self.texts = []
    
    def _get_message(self, info):
        message = info.get('message', {})
        message['RecordCallback'] = f'Recording: {"On" if self.recording else "Off"}, Recording Time: {len(self.frames)}'
        return message

    def before_reset(self, sim, reset_flag: bool) -> bool:
        if self.recording:
            self._save_episode()
            self.episode_id += 1
        return reset_flag

    def after_reset(self, sim, obs, info):
        sim.callback_messages.add("Press 'R' to start/stop recording.")
        # this message would be displayed in the GUI when command mode is on
        info['message'] = self._get_message(info)
        if self.recording:
            if self.frame_type == 'obs':
                self.frames.append(obs['image'])
            elif self.frame_type == 'pov':
                self.frames.append(info['pov'])
            else:
                raise ValueError(f'Invalid frame_type: {self.frame_type}')
            if self.record_actions:
                self.actions.append({}) #empty for reset
            if self.show_instruction:
                self.texts.append(info["task"]["text"])
            if self.record_infos:
                self.infos.append(info)
        
        return obs, info
    
    def before_step(self, sim, action):
        if self.recording and (self.record_actions  or self.show_actions):
            self.actions.append(action)
        return action
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if self.show_instruction:
            self.texts.append(info["task"]["text"])
        
        if self.recording and not info.get('R', True):
            self.recording = False
            print(f'[red]Recording stopped[/red]')
            self._save_episode()
            self.episode_id += 1

        if not self.recording and info.get('R', False):
            self.recording = True
            print(f'[green]Start recording[/green]')

        if self.recording:
            if self.frame_type == 'obs':
                self.frames.append(obs['image'])
            elif self.frame_type == 'pov':
                self.frames.append(info['pov'])
            else:
                raise ValueError(f'Invalid frame_type: {self.frame_type}')
            if self.record_infos:
                new_info = deepcopy(info)
                new_info.pop('pov')
                self.infos.append(info)
            
        info['message'] = self._get_message(info)
        
        return obs, reward, terminated, truncated, info
    
    def before_close(self, sim):
        if self.recording:
            self._save_episode()
    
    def _save_episode(self):
        if len(self.frames) == 0:
            return 
        output_path = self.record_path / f'episode_{self.episode_id}.mp4'
        
        font = cv2.FONT_HERSHEY_SIMPLEX # cv2.FONT_HERSHEY_PLAIN # cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 1
        line_type = cv2.LINE_AA
    
        if self.record_raw_observation:
            with av.open(demo_path, mode="w", format='mp4') as container:
                stream = container.add_stream("h264", rate=self.fps)
                stream.width, stream.height = self.frames[0].shape[1], self.frames[0].shape[0]

                for frame in self.frames:
                    video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

        if self.show_actions or self.show_instruction:
            demo_path = output_path.parent / ("demo_" + output_path.name)
            with av.open(demo_path, mode="w", format='mp4') as container:
                stream = container.add_stream("h264", rate=self.fps)
                stream.width, stream.height = self.frames[0].shape[1], self.frames[0].shape[0]

                for idx, frame in enumerate(self.frames):
                    # 只在需要加文字时才copy()
                    frame_with_text = frame.copy()

                    # 如果有指令文本
                    print(self.show_instruction, idx < len(self.texts))
                    if self.show_instruction and idx < len(self.texts):
                        cv2.putText(frame_with_text, self.texts[idx][:50], (20, 100), font, font_scale, font_color, thickness, line_type)

                    # 如果有动作文本
                    if self.show_actions:
                        for row, (k, v) in enumerate(self.actions[idx].items()):
                            if k in {"chat", "mobs", "voxels"}:
                                continue
                            display_v = "[{:.2f}, {:.2f}]".format(v[0], v[1]) if k == "camera" else v
                            cv2.putText(frame_with_text, f"{k}: {display_v}", (10, 25 + row * 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)

                    video_frame = av.VideoFrame.from_ndarray(frame_with_text, format="rgb24")

                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
                
        if self.record_origin_observation:
            output_origin_path = self.record_path / f'episode_{self.episode_id}.npy'
            all_frames = np.array(self.frames)
            np.save(output_origin_path, all_frames)
        
        self.frames = []
        
        if self.record_actions: # assert self.actions>0 sense self.frame > 0
            output_action_path = self.record_path / f'episode_{self.episode_id}_action.json'
            record_actions = [self._process_action(action) for action in self.actions]
            with open(output_action_path, 'w', encoding="utf-8") as file:
                json.dump(record_actions, file)
            self.actions = []
        
        if self.record_infos: # assert self.actions>0 sense self.frame > 0
            output_info_path = self.record_path / f'episode_{self.episode_id}_info.json'
            record_infos = [self._process_info(info) for info in self.infos]
            with open(output_info_path, 'w', encoding="utf-8") as file:
                json.dump(record_infos, file)
            self.infos = []
            
        print(f'[green]Episode {self.episode_id} saved at {output_path}[/green]')
        
    def forget(self):
        self.frames = []
        self.actions = []
        self.infos = []
        self.texts = []
    
        
    def _process_info(self,info:dict):
        record_info = deepcopy(info)
        if self.frame_type == 'pov':
            del record_info['pov']
        record_info = self._convert_data(record_info)
        return record_info
    
    def _process_action(self,action:spaces.Dict):
        record_action = dict(deepcopy(action))
        record_action = self._convert_data(record_action)
        return record_action
    
    def _convert_data(self,data):
        if isinstance(data, dict):
            # Iterate over items and apply conversion recursively
            return {key: self._convert_data(value) for key, value in data.items()}
        elif isinstance(data, defaultdict):
            return {key: self._convert_data(value) for key, value in data.spaces.items()}
        elif isinstance(data, spaces.Dict):
            return {key: self._convert_data(value) for key, value in data.spaces.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, DictConfig):
            # DictConfig 转换为普通字典
            return self._convert_data(dict(data))
        else:
            return data
        

    
            
        