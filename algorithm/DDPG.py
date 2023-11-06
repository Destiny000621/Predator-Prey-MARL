import torch
import torch.nn.functional as F
from collections import deque
import random
from utils.network import PolicyNetwork, DDPGQNetwork
from utils.ReplayBuffer_DDPG import ReplayBuffer_DDPG

class DDPG:
    def __init__(self, obs_dim, act_dim, hidden_size, seed, buffer_size=10000, batch_size=64):
        self.act_dim = act_dim
        self.seed = seed
        self.hidden_size = hidden_size
        self.replay_buffer = ReplayBuffer_DDPG(buffer_size, batch_size)
        
        # Define policy(actor) and Q-networks(critic)
        self.policy_net = PolicyNetwork(obs_dim, act_dim, seed, hidden_size)
        self.ddpg_q_net = DDPGQNetwork(obs_dim, act_dim, seed, hidden_size)

        # Define optimizers for policy network and Q-network
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.002, weight_decay=0.0001)
        self.ddpg_q_optimizer = torch.optim.Adam(self.ddpg_q_net.parameters(), lr=0.002, weight_decay=0.0001)

        # Define target networks
        self.target_policy_net = PolicyNetwork(obs_dim, act_dim, seed, hidden_size)
        self.target_ddpg_q_net = DDPGQNetwork(obs_dim, act_dim, seed, hidden_size)

        # Initialize target network weights to match the original networks
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_ddpg_q_net.load_state_dict(self.ddpg_q_net.state_dict())
        
    def act(self, observation):
        """Choose an action based on the policy."""
        with torch.no_grad():
            #action_probs = self.policy_net(torch.tensor(observation, dtype=torch.float32))
            #action = action_probs.argmax().item()
            action_logits = self.policy_net(torch.tensor(observation, dtype=torch.float32))
            action_probs = F.softmax(action_logits, dim=-1) # Convert logits to probabilities
            action = torch.multinomial(action_probs, 1).item() # Sample an action from the probability distribution
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self):
        """Perform DDPG learning update."""
        policy_loss = None
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return policy_loss
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Convert actions to one-hot encoded format
        actions_one_hot = torch.zeros(actions.size(0), self.act_dim)
        actions_one_hot.scatter_(1, actions.unsqueeze(-1).long(), 1)
        
        # Compute target Q-values
        next_actions = self.target_policy_net(next_states)
        target_q_values = self.target_ddpg_q_net(next_states, next_actions)
        y = rewards + (1 - dones) * 0.99 * target_q_values.squeeze()
        
        # Compute Q-network loss
        q_values = self.ddpg_q_net(states, actions_one_hot).squeeze()
        q_loss = F.mse_loss(q_values, y.detach())
        
        # Update Q-network
        self.ddpg_q_optimizer.zero_grad()
        q_loss.backward()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.ddpg_q_net.parameters(), max_norm=1.0)
        self.ddpg_q_optimizer.step()
        
        # Compute policy loss
        policy_actions = self.policy_net(states)
        policy_loss = -self.ddpg_q_net(states, policy_actions).mean()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # Soft update of the target networks
        self.soft_update(self.policy_net, self.target_policy_net, tau=0.01)
        self.soft_update(self.ddpg_q_net, self.target_ddpg_q_net, tau=0.01)

        return policy_loss
    
    @staticmethod
    def soft_update(local_model, target_model, tau=0.01):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


