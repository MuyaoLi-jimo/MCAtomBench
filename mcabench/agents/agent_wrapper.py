from typing import Literal
from mcabench.agents.vla.vla_agent import RT2AGENT
from mcabench.agents.coa.coa import CoaAgent
from mcabench.agents import base_agent

def make_agent(agent_mode:Literal["rt2","coa"]=None,**model_config)->base_agent.Agent:
    agent = None
    if agent_mode == "rt2":
        agent = RT2AGENT(**model_config)
    elif agent_mode == "coa":
        agent = CoaAgent(**model_config)
    else:
        raise AssertionError(f"agent mode-{agent_mode} unknown")
    return agent