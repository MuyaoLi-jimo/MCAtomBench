from typing import Literal
from mcabench.agents.vla.vla_agent import RT2_AGENT
from mcabench.agents import base_agent

def make_agent(agent_mode:Literal["rt2"]=None,**model_config)->base_agent.Agent:
    agent = None
    if agent_mode == "rt2":
        agent = RT2_AGENT(**model_config)
    else:
        raise AssertionError(f"agent mode-{agent_mode} unknown")
    return agent