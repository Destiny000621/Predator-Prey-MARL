from utils.load_model import load_ddpg
from algorithm.MADDPG import MADDPG
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
import wandb

save_dir = 'MADDPG_DDPG_models'

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

maddpg_agent_predator_0 = MADDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, hidden_size=128, seed=10)
maddpg_agent_predator_1 = MADDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n, hidden_size=128, seed=20)
maddpg_agent_predator_2 = MADDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n, hidden_size=128, seed=30)
maddpg_agent_prey_0 = MADDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128, seed=40)

# Load the models for each agent
load_ddpg(maddpg_agent_predator_0, 'maddpg_agent_predator_0', save_dir)
load_ddpg(maddpg_agent_predator_1, 'maddpg_agent_predator_1', save_dir)
load_ddpg(maddpg_agent_predator_2, 'maddpg_agent_predator_2', save_dir)
load_ddpg(maddpg_agent_prey_0, 'maddpg_agent_prey_0', save_dir)

def evaluate_model(num_episodes):
    total_rewards = []

    wandb.init(project='MAPP_evaluate', name='MADDPG')

    for episode in range(num_episodes):
        episode_rewards = []
        frames = []
        observations, _ = env.reset()
        while env.agents:
            actions = {}
            for agent, obs in observations.items():
                if "predator_0" in agent:
                    actions[agent] = maddpg_agent_predator_0.act(obs)
                elif "predator_1" in agent:
                    actions[agent] = maddpg_agent_predator_1.act(obs)
                elif "predator_2" in agent:
                    actions[agent] = maddpg_agent_predator_2.act(obs)
                else:
                    actions[agent] = maddpg_agent_prey_0.act(obs)

            # Take the chosen actions and observe the next state and rewards
            next_observations, rewards, infos, done, _ = env.step(actions)
            frames.append(env.render())

            episode_rewards.append(sum(rewards.values()))
            observations = next_observations
        if episode % 10 == 0:
            SimpleEnv.display_frames_as_gif(frames,episode)

        mean_one_episode_reward = sum(episode_rewards)/len(episode_rewards)
        total_rewards.append(mean_one_episode_reward)

        wandb.log({
            "Episode Reward": sum(episode_rewards)
        })

    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    wandb.finish()

evaluate_model(num_episodes=100)
env.close()
