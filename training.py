import os
import matplotlib.pyplot as plt
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
from algorithm.MADDPG import MADDPG
from algorithm.DDPG import DDPG
import wandb

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize MADDPG agent for adversaries and DDPG agent for the prey
maddpg_agent = MADDPG(obs_dim=16, act_dim=5, num_predators=3, hidden_size=128)
ddpg_agent = DDPG(obs_dim=14, act_dim=5, hidden_size=128)

# Initialize wandb
wandb.init(project='MAPP_version1', name='maddpg_predator_ddpg_prey')

NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    observations, _ = env.reset()
    episode_rewards = []
    
    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            if "predator" in agent:
                actions[agent] = maddpg_agent.act([obs])[0]
            else:
                actions[agent] = ddpg_agent.act(obs)
        
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store experiences and update
        for agent, obs in observations.items():
            reward = rewards[agent]
            next_obs = next_observations[agent]
            done = terminations[agent]
            
            if "predator" in agent:
                maddpg_agent.store_experience(obs, actions[agent], reward, next_obs, done)
                maddpg_losses = maddpg_agent.update()
            else:
                ddpg_agent.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_loss = ddpg_agent.update()

        episode_rewards.append(sum(rewards.values()))
        observations = next_observations

    # Log rewards and policy losses to wandb
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "MADDPG Policy Loss (Adversary 0)": maddpg_losses[0] if maddpg_losses and maddpg_losses[0] is not None else None,
        "MADDPG Policy Loss (Adversary 1)": maddpg_losses[1] if maddpg_losses and maddpg_losses[1] is not None else None,
        "MADDPG Policy Loss (Adversary 2)": maddpg_losses[2] if maddpg_losses and maddpg_losses[2] is not None else None,
        "DDPG Policy Loss (Good Agent)": ddpg_loss
    })

# Finish the wandb run
wandb.finish()