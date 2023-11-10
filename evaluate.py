from utils.load_model import load_dqn
import wandb
from baseline.DQN import DQNAgent
import os
from collections import deque
from multiagent.mpe.predator_prey import predator_prey

save_dir = 'DQN_models'

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

dqn_agent_predator_0 = DQNAgent(state_size=env.observation_space("predator_0").shape[0], action_size=env.action_space("predator_0").n, seed=1)
dqn_agent_predator_1 = DQNAgent(state_size=env.observation_space("predator_1").shape[0], action_size=env.action_space("predator_1").n, seed=1)
dqn_agent_predator_2 = DQNAgent(state_size=env.observation_space("predator_2").shape[0], action_size=env.action_space("predator_2").n, seed=1)
dqn_agent_prey_0 = DQNAgent(state_size=env.observation_space("prey_0").shape[0], action_size=env.action_space("prey_0").n, seed=1)

# Load the models for each agent
load_dqn(dqn_agent_predator_0, 'dqn_agent_predator_0', save_dir)
load_dqn(dqn_agent_predator_1, 'dqn_agent_predator_1', save_dir)
load_dqn(dqn_agent_predator_2, 'dqn_agent_predator_2', save_dir)
load_dqn(dqn_agent_prey_0, 'dqn_agent_prey_0', save_dir)