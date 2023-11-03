# pyright: reportGeneralTypeIssues=false
import copy
import warnings
from collections import defaultdict
from typing import Callable, Dict, Optional

from multiagent.utils import agent_selector
from multiagent.utils.env import ActionType, AECEnv, AgentID, ObsType, ParallelEnv
from multiagent.utils.wrappers import OrderEnforcingWrapper


def parallel_wrapper_fn(    env_fn: Callable) -> Callable:
    def par_fn(**kwargs):
        env = env_fn(**kwargs)
        env = aec_to_parallel_wrapper(env)
        return env

    return par_fn

def aec_to_parallel(
    aec_env: AECEnv[AgentID, ObsType, ActionType]
) -> ParallelEnv[AgentID, ObsType, ActionType]:
    """Converts an AEC environment to a Parallel environment.

    In the case of an existing Parallel environment wrapped using a `parallel_to_aec_wrapper`, this function will return the original Parallel environment.
    Otherwise, it will apply the `aec_to_parallel_wrapper` to convert the environment.
    """
    
    par_env = aec_to_parallel_wrapper(aec_env)
    return par_env

class aec_to_parallel_wrapper(ParallelEnv[AgentID, ObsType, ActionType]):
    """Converts an AEC environment into a Parallel environment."""

    def __init__(self, aec_env):
        assert aec_env.metadata.get("is_parallelizable", False)
        self.aec_env = aec_env
        self.possible_agents = aec_env.possible_agents
        self.metadata = aec_env.metadata
        self.render_mode = self.aec_env.render_mode  
        self.state_space = self.aec_env.state_space

    def observation_space(self, agent):
        return self.aec_env.observation_space(agent)

    def action_space(self, agent):
        return self.aec_env.action_space(agent)

    @property
    def unwrapped(self):
        return self.aec_env.unwrapped

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed, options=options)
        self.agents = self.aec_env.agents[:]
        observations = {
            agent: self.aec_env.observe(agent)
            for agent in self.aec_env.agents
            if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
        }

        infos = dict(**self.aec_env.infos)
        return observations, infos

    def step(self, actions):
        rewards = defaultdict(float)
        terminations = {}
        truncations = {}
        infos = {}
        observations = {}
        for agent in self.aec_env.agents:
            if agent != self.aec_env.agent_selection:
                if self.aec_env.terminations[agent] or self.aec_env.truncations[agent]:
                    raise AssertionError(
                        f"expected agent {agent} got termination or truncation agent {self.aec_env.agent_selection}. Parallel environment wrapper expects all agent death (setting an agent's self.terminations or self.truncations entry to True) to happen only at the end of a cycle."
                    )
                else:
                    raise AssertionError(
                        f"expected agent {agent} got agent {self.aec_env.agent_selection}, Parallel environment wrapper expects agents to step in a cycle."
                    )
            obs, rew, termination, truncation, info = self.aec_env.last()
            self.aec_env.step(actions[agent])
            for agent in self.aec_env.agents:
                rewards[agent] += self.aec_env.rewards[agent]

        terminations = dict(**self.aec_env.terminations)
        truncations = dict(**self.aec_env.truncations)
        infos = dict(**self.aec_env.infos)
        observations = {
            agent: self.aec_env.observe(agent) for agent in self.aec_env.agents
        }
        while self.aec_env.agents and (
            self.aec_env.terminations[self.aec_env.agent_selection]
            or self.aec_env.truncations[self.aec_env.agent_selection]
        ):
            self.aec_env.step(None)

        self.agents = self.aec_env.agents
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.aec_env.render()

    def state(self):
        return self.aec_env.state()

    def close(self):
        return self.aec_env.close()
