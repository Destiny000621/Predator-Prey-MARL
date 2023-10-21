import torch
import torch.nn.functional as F
from collections import deque
import random
from utils.network import PolicyNetwork, MADDPGQNetwork
from utils.ReplayBuffer import ReplayBuffer

class MADDPG:
    def __init__(self, obs_dim, act_dim, num_predators, hidden_size=128, buffer_size=10000, batch_size=64):
        self.num_predators = num_predators
        self.act_dim = act_dim
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        
        # Define policy and Q-networks for each predator
        self.predator_policy_nets = [PolicyNetwork(obs_dim, act_dim, hidden_size) for _ in range(num_predators)]
        self.predator_q_net = MADDPGQNetwork(obs_dim, act_dim, num_predators, hidden_size)

        # Define optimizers for each predator's policy network and the shared Q-network
        self.predator_policy_optimizers = [torch.optim.Adam(net.parameters(), lr=0.001) for net in self.predator_policy_nets]
        self.predator_q_optimizer = torch.optim.Adam(self.predator_q_net.parameters(), lr=0.001)

        # Define target networks for each predator and the shared Q-network
        self.target_predator_policy_nets = [PolicyNetwork(obs_dim, act_dim, hidden_size) for _ in range(num_predators)]
        self.target_predator_q_net = MADDPGQNetwork(obs_dim, act_dim, num_predators, hidden_size)

        # Initialize target network weights to match the original networks
        for i in range(num_predators):
            self.target_predator_policy_nets[i].load_state_dict(self.predator_policy_nets[i].state_dict())
        self.target_predator_q_net.load_state_dict(self.predator_q_net.state_dict())
        
    def act(self, observations):
        """Choose actions for all predators based on their policies."""
        actions = []
        with torch.no_grad():
            for i, obs in enumerate(observations):
                action_probs = self.predator_policy_nets[i](torch.tensor(obs, dtype=torch.float32))
                action = action_probs.argmax().item()
                actions.append(action)
        return actions
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self):
        """Perform MADDPG learning update for predators."""
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Convert actions to one-hot encoded format and get next actions
        actions_one_hot = torch.zeros(actions.size(0), self.act_dim)
        actions_one_hot.scatter_(1, actions.unsqueeze(-1).long(), 1)
        next_actions = torch.stack([net(next_states) for net in self.predator_policy_nets], dim=1).view(actions.size(0), -1)
        
        # Compute target Q-values
        target_q_values = self.target_predator_q_net(next_states.repeat(1, self.num_predators), next_actions)
        y = rewards + (1 - dones) * 0.99 * target_q_values.squeeze()
        
        # Compute Q-network loss
        q_values = self.predator_q_net(states.repeat(1, self.num_predators), actions_one_hot.repeat(1, self.num_predators)).squeeze()
        q_loss = F.mse_loss(q_values, y.detach())
        
        # Update Q-network
        self.predator_q_optimizer.zero_grad()
        q_loss.backward()
        self.predator_q_optimizer.step()
        
        # Update each predator's policy network
        for i, (policy_net, optimizer) in enumerate(zip(self.predator_policy_nets, self.predator_policy_optimizers)):
            # Compute policy loss
            curr_actions = policy_net(states)
            other_actions = torch.cat([self.predator_policy_nets[j](states) if j != i else curr_actions for j in range(self.num_predators)], dim=1)
            policy_loss = -self.predator_q_net(states.repeat(1, self.num_predators), other_actions).mean()
            
            # Update policy network
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
        
        # Soft update of the target networks
        for policy_net, target_policy_net in zip(self.predator_policy_nets, self.target_predator_policy_nets):
            self.soft_update(policy_net, target_policy_net, tau=0.01)
        self.soft_update(self.predator_q_net, self.target_predator_q_net, tau=0.01)
    
    @staticmethod
    def soft_update(local_model, target_model, tau=0.01):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Instantiate the MADDPG class for demonstration
OBS_DIM = 16
ACT_DIM = 5
num_predator = 3
maddpg_agent = MADDPG(obs_dim=OBS_DIM, act_dim=ACT_DIM, num_predators=num_predator, hidden_size=128)
maddpg_agent