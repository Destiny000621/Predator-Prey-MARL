#TODO: Add more parameters to the wandb log later such loss, etc.
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import wandb

# Initialize a new wandb run
wandb.init(project="predator_prey", name="random_policy")
num_episodes = 50

for episode in range(num_episodes):

    env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
    observations, infos = env.reset()

    cumulative_rewards = {agent: 0 for agent in env.agents}

    while env.agents:
        # insert policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update cumulative rewards
        for agent, reward in rewards.items():
            cumulative_rewards[agent] += reward

    wandb.log({"episode": episode, **cumulative_rewards})
    env.close()

wandb.finish()