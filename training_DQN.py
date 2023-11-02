import wandb
from DQN import DQNAgent # Import your DQN agent here
import time
from collections import deque
from multiagent.mpe.predator_prey import predator_prey

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize DQN agents
dqn_agent_predator_0 = DQNAgent(state_size=env.observation_space("predator_0").shape[0], action_size=env.action_space("predator_0").n, seed=1)
dqn_agent_predator_1 = DQNAgent(state_size=env.observation_space("predator_1").shape[0], action_size=env.action_space("predator_1").n, seed=1)
dqn_agent_predator_2 = DQNAgent(state_size=env.observation_space("predator_2").shape[0], action_size=env.action_space("predator_2").n, seed=1)
dqn_agent_prey_0 = DQNAgent(state_size=env.observation_space("prey_0").shape[0], action_size=env.action_space("prey_0").n, seed=1)

# Initialize wandb
wandb.init(project='MAPP_version1', name='DQN')

# Define the number of episodes and epsilon for exploration
NUM_EPISODES = 5000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.999
eps = EPS_START

# Define a window for averaging episode rewards
WINDOW_SIZE = 500
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

for episode in range(NUM_EPISODES):
    observations, _ = env.reset()
    episode_rewards = []

    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            # Select an action using the epsilon-greedy policy
            if "predator_0" in agent:
                actions[agent] = dqn_agent_predator_0.act(obs, eps)
            elif "predator_1" in agent:
                actions[agent] = dqn_agent_predator_1.act(obs, eps)
            elif "predator_2" in agent:
                actions[agent] = dqn_agent_predator_2.act(obs, eps)
            else:
                actions[agent] = dqn_agent_prey_0.act(obs, eps)

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store experiences and update
        for agent, obs in observations.items():
            action = actions[agent]
            reward = rewards[agent]
            next_obs = next_observations[agent]
            done = terminations[agent]

            if "predator_0" in agent:
                dqn_agent_predator_0.step(obs, action, reward, next_obs, done)
            elif "predator_1" in agent:
                dqn_agent_predator_1.step(obs, action, reward, next_obs, done)
            elif "predator_2" in agent:
                dqn_agent_predator_2.step(obs, action, reward, next_obs, done)
            else:
                dqn_agent_prey_0.step(obs, action, reward, next_obs, done)

        episode_rewards.append(sum(rewards.values()))
        observations = next_observations

    # Update epsilon
    eps = max(EPS_END, EPS_DECAY * eps)

    # Append the cumulative reward of this episode to the window
    episode_rewards_window.append(sum(episode_rewards))

    # Compute the mean reward over the last WINDOW_SIZE episodes
    mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)

    # Log rewards and other metrics to wandb
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "Mean Episode Reward (Last {} episodes)".format(WINDOW_SIZE): mean_episode_reward,
        "Epsilon": eps
    })

# Finish the wandb run
wandb.finish()