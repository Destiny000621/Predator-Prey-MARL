from utils.load_model import load_dqn
from baseline.DQN import DQNAgent
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np


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

# ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, hidden_size=128, seed=10)
# ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n, hidden_size=128, seed=20)
# ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n, hidden_size=128, seed=30)
# ddpg_agent_prey_0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128, seed=40)

# Load the models for each agent
# load_ddpg(ddpg_agent_predator_0, 'ddpg_agent_predator_0', save_dir)
# load_ddpg(ddpg_agent_predator_1, 'ddpg_agent_predator_1', save_dir)
# load_ddpg(ddpg_agent_predator_2, 'ddpg_agent_predator_2', save_dir)
# load_ddpg(ddpg_agent_prey_0, 'ddpg_agent_prey_0', save_dir)

def evaluate_model(num_episodes=10):
    total_rewards = []

    for episode in range(num_episodes):
        episode_rewards = []
        frames = []
        observations, _ = env.reset()
        while env.agents:
            actions = {}
            # actions = act(obs)
            for agent, obs in observations.items():
                # if "predator_0" in agent:
                #     actions[agent] = ddpg_agent_predator_0.act(obs)
                # elif "predator_1" in agent:
                #     actions[agent] = ddpg_agent_predator_1.act(obs)
                # elif "predator_2" in agent:
                #     actions[agent] = ddpg_agent_predator_2.act(obs)
                # else:
                #     actions[agent] = ddpg_agent_prey_0.act(obs)

                if "predator_0" in agent:
                    actions[agent] = dqn_agent_predator_0.act(obs)
                elif "predator_1" in agent:
                    actions[agent] = dqn_agent_predator_1.act(obs)
                elif "predator_2" in agent:
                    actions[agent] = dqn_agent_predator_2.act(obs)
                else:
                    actions[agent] = dqn_agent_prey_0.act(obs)

            # Take the chosen actions and observe the next state and rewards
            next_observations, rewards, infos, done, _ = env.step(actions)
            frames.append(env.render())

            episode_rewards.append(sum(rewards.values()))
            observations = next_observations
        SimpleEnv.display_frames_as_gif(frames,episode)

        total_rewards.append(episode_rewards)

    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')

    # if save_gif_path:
    #     imageio.mimsave(save_gif_path, frames, duration=0.1)

# gif_path = '/evaluation.gif'
evaluate_model(num_episodes=10)
env.close()