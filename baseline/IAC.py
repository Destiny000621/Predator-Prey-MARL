import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Constants
HIDDEN_SIZE = 64

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, seed, HIDDEN_SIZE):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_dim)

    def forward(self, x):
        # Define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class Critic(nn.Module):
    def __init__(self, input_dim, seed, HIDDEN_SIZE):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Define your network layers here
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        # Define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class IACAgent:
    def __init__(self, state_size, action_size, seed, learning_rate=1e-3):
        self.actor = Actor(state_size, action_size, seed)
        self.critic = Critic(state_size, seed)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.actor(state).cpu()
        # Sample an action from the output probabilities
        action = torch.multinomial(probs, 1).item()
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        action = torch.tensor(action).view(-1, 1)
        reward = torch.tensor(reward).float().view(-1, 1)
        done = torch.tensor(done).float().view(-1, 1)

        # Get predicted current state value and next state value
        V_curr = self.critic(state)
        V_next = self.critic(next_state).detach()

        # Compute the target and advantage
        target = reward + (1 - done) * V_next
        advantage = target - V_curr

        # Update critic to minimize value loss
        value_loss = F.mse_loss(V_curr, target)
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()

        # Update actor to maximize policy objective function using advantage
        probs = self.actor(state)
        # Negative sign because we want to do gradient ascent but optimizers in PyTorch do descent
        policy_loss = -torch.log(probs.gather(1, action)) * advantage.detach()
        self.optimizer_actor.zero_grad()
        policy_loss.mean().backward()
        self.optimizer_actor.step()

# if __name__ == '__main__':
# # Initialize IAC agents
#     iac_agent_predator_0 = IACAgent(state_size=env.observation_space("predator_0").shape[0],
#                                     action_size=env.action_space("predator_0").n, seed=1)
#     iac_agent_predator_1 = IACAgent(state_size=env.observation_space("predator_1").shape[0],
#                                     action_size=env.action_space("predator_1").n, seed=1)
#     iac_agent_predator_2 = IACAgent(state_size=env.observation_space("predator_2").shape[0],
#                                     action_size=env.action_space("predator_2").n, seed=1)
#     iac_agent_prey_0 = IACAgent(state_size=env.observation_space("prey_0").shape[0],
#                                 action_size=env.action_space("prey_0").n, seed=1)
#
#     # Training loop setup remains the same
#     for episode in range(NUM_EPISODES):
#         # Rest of the code remains the same up to the for loop for interactions with environment
#
#         for agent, obs in observations.items():
#             # Select an action using the actor network
#             if "predator" in agent:
#                 action = iac_agents[agent].act(obs)
#             else:
#                 action = iac_agents[agent].act(obs)
#             actions[agent] = action
#
#         # Rest of the interaction loop remains the same
#
#         # Learning step
#         for agent, obs in observations.items():
#             action = actions[agent]
#             reward = rewards[agent]
#             next_obs = next_observations[agent]
#             done = terminations[agent] or truncations[agent]
#
#             # Agents learn from their experiences
#             if "predator" in agent:
#                 iac_agents[agent].learn(obs, action, reward, next_obs, done)
#             else:
#                 iac_agents[agent].learn(obs, action, reward, next_obs, done)
#
#         # Rest of the training loop remains the same