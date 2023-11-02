from multiagent.mpe.predator_prey import predator_prey
from algorithm.MADDPG import MADDPG
from algorithm.DDPG import DDPG
import wandb
from collections import deque
import random

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize MADDPG agent for predators and DDPG agent for the prey
maddpg_agent = MADDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, num_predators = 3, hidden_size=128)
ddpg_agent = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128)

# Initialize wandb
#t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
wandb.init(project='MAPP_version1', name='MADDPG')

# Define the episode length
NUM_EPISODES = 3000

# Define a window size for averaging episode rewards
WINDOW_SIZE = 200
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

for episode in range(NUM_EPISODES):
    observations, _ = env.reset()
    episode_rewards = []
    
    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            if "predator" in agent:
                actions[agent] = maddpg_agent.act([obs])[0]
            else:
                actions[agent] = action = random.randint(0, 4)
        
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store experiences and update
        for agent, obs in observations.items():
            if "predator" in agent:
                reward = rewards[agent]
                next_obs = next_observations[agent]
                done = terminations[agent]

                maddpg_agent.store_experience(obs, actions[agent], reward, next_obs, done)
                maddpg_losses = maddpg_agent.update()

        episode_rewards.append(sum(rewards.values()))
        observations = next_observations

    # Append the episode's cumulative reward to the window
    episode_rewards_window.append(sum(episode_rewards))

    # Compute the mean reward over the last N episodes
    mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)

    # Log rewards and policy losses to wandb
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "Mean Episode Reward (Last 200 episodes)": mean_episode_reward,
        "MADDPG Policy Loss (Predator 0)": maddpg_losses[0] if maddpg_losses and maddpg_losses[0] is not None else None,
        "MADDPG Policy Loss (Predator 1)": maddpg_losses[1] if maddpg_losses and maddpg_losses[1] is not None else None,
        "MADDPG Policy Loss (Predator 2)": maddpg_losses[2] if maddpg_losses and maddpg_losses[2] is not None else None,
    })

# Finish the wandb run
wandb.finish()