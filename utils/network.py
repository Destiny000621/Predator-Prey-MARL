import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
OBS_DIM_Predator = 16
OBS_DIM_Prey =14
ACT_DIM = 5
HIDDEN_SIZE = 128
num_predators = 3

# Define the policy network(MLP)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, HIDDEN_SIZE):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)  # Return a probability distribution over actions

# Define the Q-network for MADDPG
class MADDPGQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, HIDDEN_SIZE):
        super(MADDPGQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim * num_agents + act_dim * num_agents, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)  # Outputs a single Q-value

    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the Q-network for DDPG
class DDPGQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, HIDDEN_SIZE):
        super(DDPGQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)  # Outputs a single Q-value

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Create instances of the networks for demonstration
policy_net_predator = PolicyNetwork(OBS_DIM_Predator, ACT_DIM, HIDDEN_SIZE)
policy_net_prey = PolicyNetwork(OBS_DIM_Prey, ACT_DIM, HIDDEN_SIZE)
maddpg_q_net = MADDPGQNetwork(OBS_DIM_Predator, ACT_DIM, num_predators, HIDDEN_SIZE)
ddpg_q_net = DDPGQNetwork(OBS_DIM_Prey, ACT_DIM, HIDDEN_SIZE)

#print(policy_net_predator, policy_net_prey, maddpg_q_net, ddpg_q_net)
