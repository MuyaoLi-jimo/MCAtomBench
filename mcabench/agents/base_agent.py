import abc
from minestudio.simulator.entry import MinecraftSim
from minestudio.simulator.callbacks.callback import MinecraftCallback

class Agent(abc.ABC):
    def __init__(self,**kwargs):
        self._action_type = "agent"
    
    @abc.abstractmethod
    def get_instructions(self,env,env_cfg):
        pass
    
    def get_observations(self,env,info:dict):
        return [info["pov"]]
    
    def show(self,record_callback:MinecraftCallback,):
        pass
    
    @abc.abstractmethod
    def forward(self,observations:list,instructions:list,verbos=False):
        pass
        
    @abc.abstractmethod
    def reset(self,env:MinecraftSim):
        pass
    
    @property
    def action_type(self):
        return self._action_type