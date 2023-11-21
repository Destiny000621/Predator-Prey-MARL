from utils.load_model import load_ddpg
from algorithm.DDPG import DDPG
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
import wandb
import os

save_dir = 'DDPG_DDPG_models'

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, hidden_size=128, seed=10)
ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n, hidden_size=128, seed=20)
ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n, hidden_size=128, seed=30)
ddpg_agent_prey_0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128, seed=40)

# Load the models for each agent
load_ddpg(ddpg_agent_predator_0, 'ddpg_agent_predator_0', save_dir)
load_ddpg(ddpg_agent_predator_1, 'ddpg_agent_predator_1', save_dir)
load_ddpg(ddpg_agent_predator_2, 'ddpg_agent_predator_2', save_dir)
load_ddpg(ddpg_agent_prey_0, 'ddpg_agent_prey_0', save_dir)

# Set a folder to save the gifs
gif_dir = 'DDPG_gifs'
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)

def evaluate_model(num_episodes):
    total_rewards = []

    wandb.init(project='MAPP_evaluate', name='DDPG')

    for episode in range(num_episodes):
        episode_rewards = []
        frames = []
        observations, _ = env.reset()
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
                    actions[agent] = ddpg_agent_prey_0.act(obs)

            # Take the chosen actions and observe the next state and rewards
            next_observations, rewards, terminations, infos, _ = env.step(actions)

            # Store experiences and update
            for agent, obs in observations.items():
                reward = rewards[agent]
                next_obs = next_observations[agent]
                done = terminations[agent]

                if "predator_0" in agent:
                    ddpg_agent_predator_0.store_experience(obs, actions[agent], reward, next_obs, done)
                elif "predator_1" in agent:
                    ddpg_agent_predator_1.store_experience(obs, actions[agent], reward, next_obs, done)
                elif "predator_2" in agent:
                    ddpg_agent_predator_2.store_experience(obs, actions[agent], reward, next_obs, done)
                else:
                    ddpg_agent_prey_0.store_experience(obs, actions[agent], reward, next_obs, done)

            frames.append(env.render())

            episode_rewards.append(sum(rewards.values()))
            observations = next_observations

        if episode % 20 == 0:
            SimpleEnv.display_frames_as_gif(frames, episode, gif_dir)

        mean_one_episode_reward = sum(episode_rewards)/len(episode_rewards)
        total_rewards.append(mean_one_episode_reward)

        wandb.log({
            "Mean Episode Reward": mean_one_episode_reward
        })

    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    wandb.finish()

evaluate_model(num_episodes=200)
env.close()