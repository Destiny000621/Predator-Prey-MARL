from utils.load_model import load_dqn
from baseline.DQN import DQNAgent
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
import wandb
import os

save_dir = 'DQN_models'

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

dqn_agent_predator_0 = DQNAgent(state_size=env.observation_space("predator_0").shape[0], action_size=env.action_space("predator_0").n, seed=11)
dqn_agent_predator_1 = DQNAgent(state_size=env.observation_space("predator_1").shape[0], action_size=env.action_space("predator_1").n, seed=22)
dqn_agent_predator_2 = DQNAgent(state_size=env.observation_space("predator_2").shape[0], action_size=env.action_space("predator_2").n, seed=33)
dqn_agent_prey_0 = DQNAgent(state_size=env.observation_space("prey_0").shape[0], action_size=env.action_space("prey_0").n, seed=44)

# Load the models for each agent
load_dqn(dqn_agent_predator_0, 'dqn_agent_predator_0', save_dir)
load_dqn(dqn_agent_predator_1, 'dqn_agent_predator_1', save_dir)
load_dqn(dqn_agent_predator_2, 'dqn_agent_predator_2', save_dir)
load_dqn(dqn_agent_prey_0, 'dqn_agent_prey_0', save_dir)

# Initialize wandb
wandb.init(project='MAPP_evaluate', name='DQN')

# Set a folder to save the gifs
gif_dir = 'DQN_gifs'
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)

eps = 0.01
def evaluate_model(num_episodes):
    total_rewards = []

    wandb.init(project='MAPP_evaluate', name='DQN')

    for episode in range(num_episodes):
        episode_rewards = []
        frames = []
        observations, _ = env.reset()
        while env.agents:
            actions = {}
            # actions = act(obs)
            for agent, obs in observations.items():
                if "predator_0" in agent:
                    actions[agent] = dqn_agent_predator_0.act(obs, eps)
                elif "predator_1" in agent:
                    actions[agent] = dqn_agent_predator_1.act(obs, eps)
                elif "predator_2" in agent:
                    actions[agent] = dqn_agent_predator_2.act(obs, eps)
                else:
                    actions[agent] = dqn_agent_prey_0.act(obs, eps)

            # Take the chosen actions and observe the next state and rewards
            frames.append(env.render())
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

        if episode % 20 == 0:
            SimpleEnv.display_frames_as_gif(frames, episode, gif_dir)

        mean_one_episode_reward = sum(episode_rewards) / len(episode_rewards)
        total_rewards.append(mean_one_episode_reward)
        mean_one_episode_reward = sum(episode_rewards) / len(episode_rewards)
        total_rewards.append(mean_one_episode_reward)

        wandb.log({
            "Mean Episode Reward": mean_one_episode_reward
        })

    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    wandb.finish()

evaluate_model(num_episodes=200)
env.close()