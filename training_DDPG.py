import os
import matplotlib.pyplot as plt
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
from algorithm.DDPG import DDPG
import wandb
import time
from collections import deque

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize DDPG agent for predators and the prey
ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, hidden_size=128)
ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n, hidden_size=128)
ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n, hidden_size=128)
ddpg_agent = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128)

# Initialize wandb
#t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
wandb.init(project='MAPP_version1', name='DDPG-DDPG')

# Define the episode length
NUM_EPISODES = 2000

# Define a window size for averaging episode rewards
WINDOW_SIZE = 500
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

for episode in range(NUM_EPISODES):
    observations, _ = env.reset()
    episode_rewards = []
    
    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            if "predator_0" in agent:
                actions[agent] = ddpg_agent_predator_0.act(obs)
            elif "predator_1" in agent:
                actions[agent] = ddpg_agent_predator_1.act(obs)
            elif "predator_2" in agent:
                actions[agent] = ddpg_agent_predator_2.act(obs)
            else:
                actions[agent] = ddpg_agent.act(obs)
        
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store experiences and update
        for agent, obs in observations.items():
            reward = rewards[agent]
            next_obs = next_observations[agent]
            done = terminations[agent]
            
            if "predator_0" in agent:
                ddpg_agent_predator_0.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_losses_0 = ddpg_agent_predator_0.update()
            elif "predator_1" in agent:
                ddpg_agent_predator_1.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_losses_1 = ddpg_agent_predator_1.update()
            elif "predator_2" in agent:
                ddpg_agent_predator_2.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_losses_2 = ddpg_agent_predator_2.update()
            else:
                ddpg_agent.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_loss = ddpg_agent.update()

        episode_rewards.append(sum(rewards.values()))
        observations = next_observations

    # Append the episode's cumulative reward to the window
    episode_rewards_window.append(sum(episode_rewards))

    # Compute the mean reward over the last N episodes
    mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)

    # Log rewards and policy losses to wandb
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "Mean Episode Reward (Last {} episodes)".format(WINDOW_SIZE): mean_episode_reward,
        "DDPG Policy Loss (Predator 0)": ddpg_losses_0,
        "DDPG Policy Loss (Predator 1)": ddpg_losses_1,
        "DDPG Policy Loss (Predator 2)": ddpg_losses_2,
        "DDPG Policy Loss (Prey 0)": ddpg_loss
    })

# Finish the wandb run
wandb.finish()