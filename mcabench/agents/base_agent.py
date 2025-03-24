import abc

class Agent(abc.ABC):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._action_type = "agent"
        pass
    
    @abc.abstractmethod
    def get_instructions(self,env,env_cfg):
        pass
    
    def get_observations(self,env,info:dict):
        return [info["pov"]]
    
    @abc.abstractmethod
    def forward(self,observations:list,instructions:list,verbos=False):
        pass
        
    @abc.abstractmethod
    def reset(self):
        pass
    
    @property
    def action_type(self):
        return self._action_type