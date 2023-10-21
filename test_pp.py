import os
import matplotlib.pyplot as plt
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np

# save images and GIFs
# default setting: num_prey=1, num_predators=3, num_obstacles=2, max_cycles=25
env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

frames = []  # List to store images for every step

save_dir = "predator_prey_step"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

step_count = 0

while env.agents:
    # Capture the environment state as an image
    frame = env.render()
    frames.append(frame)
    plt.imsave(os.path.join(save_dir, f"pp_{step_count:04d}.png"), frame)
    step_count += 1

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    #agents' self velocities
    velocities = {agent: observation[0:2] for agent, observation in observations.items()}
    speeds = {agent: np.sqrt(np.square(vel[0]) + np.square(vel[1])) for agent, vel in velocities.items()}
    #agents' self positions
    self_positions = {agent: observation[2:4] for agent, observation in observations.items()}

SimpleEnv.display_frames_as_gif(frames)
env.close()