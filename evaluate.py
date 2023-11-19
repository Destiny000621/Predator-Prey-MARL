from utils.load_model import load_ddpg
import wandb
from baseline.DQN import DQNAgent
from algorithm.DDPG import DDPG
import os
from collections import deque
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
import imageio


save_dir = 'DDPG_DDPG_models'

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
# env_ = SimpleEnv(scenario=env.scenario, world=env.world, render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# dqn_agent_predator_0 = DQNAgent(state_size=env.observation_space("predator_0").shape[0], action_size=env.action_space("predator_0").n, seed=1)
# dqn_agent_predator_1 = DQNAgent(state_size=env.observation_space("predator_1").shape[0], action_size=env.action_space("predator_1").n, seed=1)
# dqn_agent_predator_2 = DQNAgent(state_size=env.observation_space("predator_2").shape[0], action_size=env.action_space("predator_2").n, seed=1)
# dqn_agent_prey_0 = DQNAgent(state_size=env.observation_space("prey_0").shape[0], action_size=env.action_space("prey_0").n, seed=1)
#
# # Load the models for each agent
# load_dqn(dqn_agent_predator_0, 'dqn_agent_predator_0', save_dir)
# load_dqn(dqn_agent_predator_1, 'dqn_agent_predator_1', save_dir)
# load_dqn(dqn_agent_predator_2, 'dqn_agent_predator_2', save_dir)
# load_dqn(dqn_agent_prey_0, 'dqn_agent_prey_0', save_dir)

ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, hidden_size=128, seed=10)
ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n, hidden_size=128, seed=20)
ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n, hidden_size=128, seed=30)
ddpg_agent_prey_0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128, seed=40)

# Load the models for each agent
load_ddpg(ddpg_agent_predator_0, 'ddpg_agent_predator_0', save_dir)
load_ddpg(ddpg_agent_predator_1, 'ddpg_agent_predator_1', save_dir)
load_ddpg(ddpg_agent_predator_2, 'ddpg_agent_predator_2', save_dir)
load_ddpg(ddpg_agent_prey_0, 'ddpg_agent_prey_0', save_dir)

def evaluate_model(num_episodes=10):
    total_rewards = []

    for episode in range(num_episodes):
        episode_rewards = []
        observations, _ = env.reset()
        while env.agents:
            actions = {}
            # actions = act(obs)
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
            next_observations, rewards, infos, done, _ = env.step(actions)
            frames = []
            frames.append(env.render())

            episode_rewards.append(sum(rewards.values()))
            observations = next_observations

        total_rewards.append(episode_rewards)

    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')

    imageio.mimsave('/evaluation.gif', frames, duration=0.1)


# Call the evaluation function
evaluate_model(num_episodes=10)

# def run(config):
#     model_path = (Path('./models') / config.env_id / config.model_name /
#                   ('run%i' % config.run_num))
#     if config.incremental is not None:
#         model_path = model_path / 'incremental' / ('model_ep%i.pt' %
#                                                    config.incremental)
#     else:
#         model_path = model_path / 'model.pt'
#
#     if config.save_gifs:
#         gif_path = model_path.parent / 'gifs'
#         gif_path.mkdir(exist_ok=True)
#
#     maddpg = MADDPG.init_from_save(model_path)
#     env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
#     # env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
#     maddpg.prep_rollouts(device='cpu')
#     ifi = 1 / config.fps  # inter-frame interval
#
#     for ep_i in range(config.n_episodes):
#         print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
#         obs = env.reset()
#         if config.save_gifs:
#             frames = []
#             frames.append(env.render('rgb_array')[0])
#         env.render('human')
#         for t_i in range(config.episode_length):
#             calc_start = time.time()
#             # rearrange observations to be per agent, and convert to torch Variable
#             torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
#                                   requires_grad=False)
#                          for i in range(maddpg.nagents)]
#             # get actions as torch Variables
#             torch_actions = maddpg.step(torch_obs, explore=False)
#             # convert actions to numpy arrays
#             actions = [ac.data.numpy().flatten() for ac in torch_actions]
#             obs, rewards, dones, infos = env.step(actions)
#             if config.save_gifs:
#                 frames.append(env.render('rgb_array')[0])
#             calc_end = time.time()
#             elapsed = calc_end - calc_start
#             if elapsed < ifi:
#                 time.sleep(ifi - elapsed)
#             env.render('human')
#         if config.save_gifs:
#             gif_num = 0
#             while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
#                 gif_num += 1
#             imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
#                             frames, duration=ifi)
#
#     env.close()
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("env_id", help="Name of environment")
#     parser.add_argument("model_name",
#                         help="Name of model")
#     parser.add_argument("run_num", default=1, type=int)
#     parser.add_argument("--save_gifs", action="store_true",
#                         help="Saves gif of each episode into model directory")
#     parser.add_argument("--incremental", default=None, type=int,
#                         help="Load incremental policy from given episode " +
#                              "rather than final policy")
#     parser.add_argument("--n_episodes", default=10, type=int)
#     parser.add_argument("--episode_length", default=25, type=int)
#     parser.add_argument("--fps", default=30, type=int)
#
#     config = parser.parse_args()
#
#     run(config)