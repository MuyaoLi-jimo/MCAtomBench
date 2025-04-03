from typing import Literal
from mcabench.agents.vla.vla_agent import RT2AGENT
from mcabench.agents.coa.coa import CoaAgent,LatentCoaAgent,RawActionCoaAgent
from mcabench.agents import base_agent

def make_agent(agent_mode:Literal["rt2","coa","raw-action-coa","latent-coa"]=None,**model_config)->base_agent.Agent:
    agent = None
    if agent_mode == "rt2":
        agent = RT2AGENT(**model_config)
    elif agent_mode == "coa" or agent_mode == "raw-action-coa":
        agent = RawActionCoaAgent(**model_config)
    elif agent_mode == "latent-coa":
        agent = RawActionCoaAgent(**model_config)
    else:
        raise AssertionError(f"agent mode-{agent_mode} unknown")
    return agent