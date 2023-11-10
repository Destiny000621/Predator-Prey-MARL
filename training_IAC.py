import wandb
from baseline.IAC import IACAgent
import os
from collections import deque
from multiagent.mpe.predator_prey import predator_prey
from utils.save_model import save_iac

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize IAC agents
iac_agent_predator_0 = IACAgent(state_size=env.observation_space("predator_0").shape[0], action_size=env.action_space("predator_0").n, seed=10, HIDDEN_SIZE=128)
iac_agent_predator_1 = IACAgent(state_size=env.observation_space("predator_1").shape[0], action_size=env.action_space("predator_1").n, seed=20, HIDDEN_SIZE=128)
iac_agent_predator_2 = IACAgent(state_size=env.observation_space("predator_2").shape[0], action_size=env.action_space("predator_2").n, seed=30, HIDDEN_SIZE=128)
iac_agent_prey_0 = IACAgent(state_size=env.observation_space("prey_0").shape[0], action_size=env.action_space("prey_0").n, seed=40, HIDDEN_SIZE=128)

# Initialize wandb
wandb.init(project='MAPP_version1', name='IAC')

# Define the number of episodes and epsilon for exploration
NUM_EPISODES = 15000
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY = 0.99
eps = EPS_START

# Define a window for averaging episode rewards
WINDOW_SIZE = 1000
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

# Define the path where you want to save the models
save_dir = 'IAC_models'  # Make sure this directory exists or create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for episode in range(NUM_EPISODES):
    observations, _ = env.reset()
    episode_rewards = []

    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            # Select an action using the epsilon-greedy policy
            if "predator_0" in agent:
                actions[agent] = iac_agent_predator_0.act(obs)
            elif "predator_1" in agent:
                actions[agent] = iac_agent_predator_1.act(obs)
            elif "predator_2" in agent:
                actions[agent] = iac_agent_predator_2.act(obs)
            else:
                actions[agent] = iac_agent_prey_0.act(obs)

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store experiences and update
        for agent, obs in observations.items():
            action = actions[agent]
            reward = rewards[agent]
            next_obs = next_observations[agent]
            done = terminations[agent]

            if "predator_0" in agent:
                iac_agent_predator_0.learn(obs, action, reward, next_obs, done)
            elif "predator_1" in agent:
                iac_agent_predator_1.learn(obs, action, reward, next_obs, done)
            elif "predator_2" in agent:
                iac_agent_predator_2.learn(obs, action, reward, next_obs, done)
            else:
                iac_agent_prey_0.learn(obs, action, reward, next_obs, done)

        episode_rewards.append(sum(rewards.values()))
        observations = next_observations

    # Calculate the mean reward for each episode by averaging over all the steps
    mean_one_episode_reward = sum(episode_rewards)/len(episode_rewards)

    # Update epsilon
    eps = max(EPS_END, EPS_DECAY * eps)

    # Append the cumulative reward of this episode to the window
    episode_rewards_window.append(sum(episode_rewards))

    # Compute the mean reward over the last WINDOW_SIZE episodes
    mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)

    # Log rewards and other metrics to wandb
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "Mean Episode Reward": mean_one_episode_reward,
        "Mean Episode Reward (Last {} episodes)".format(WINDOW_SIZE): mean_episode_reward,
        #"Epsilon": eps
    })
    # Save the models in the last episode
    if episode == NUM_EPISODES - 1:
        save_iac(iac_agent_predator_0, 'iac_agent_predator_0', save_dir)
        save_iac(iac_agent_predator_1, 'iac_agent_predator_1', save_dir)
        save_iac(iac_agent_predator_2, 'iac_agent_predator_2', save_dir)
        save_iac(iac_agent_prey_0, 'iac_agent_prey_0', save_dir)

# Finish the wandb run
wandb.finish()