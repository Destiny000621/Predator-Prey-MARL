import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

# Hyperparameters
LR = 0.001                 # learning rate
BUFFER_SIZE = 100000    # replay buffer size
BATCH_SIZE = 64           # minibatch size
GAMMA = 0.99              # discount factor
TAU = 0.001                # for soft update of target parameters
UPDATE_EVERY = 1          # how often to update the network
HIDDEN_SIZE = 128         # hidden layer size

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, seed, HIDDEN_SIZE):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer_DQN:
    def __init__(self, action_size, buffer_size, batch_size, seed, capacity, alpha=0.6):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.alpha = alpha  # degree of prioritization, 0 for uniform, 1 for fully prioritized
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0  # Initialize the position attribute

    def __len__(self):
        return len(self.buffer) 
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        max_priority = self.priorities.max() if self.memory else 1.0
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity  # Update the position
    
    def sample(self):
        # Proportional sampling based on TD error
        if len(self.memory) == self.memory.maxlen:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        experiences = [self.memory[i] for i in indices]
        #experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        #return (states, actions, rewards, next_states, dones)
        return (states, actions, rewards, next_states, dones), indices
    
    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = error + 1e-5  # Avoid zero probability

class DQNAgent:
    def __init__(self, state_size, action_size, seed, hidden_size=HIDDEN_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.hidden_size = hidden_size
        
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR, weight_decay=0.0001)
        
        self.memory = ReplayBuffer_DQN(action_size, BUFFER_SIZE, BATCH_SIZE, seed, capacity=100000)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values).item()
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
