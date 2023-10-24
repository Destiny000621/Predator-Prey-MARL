from collections import deque
import random
import torch

# Replay Buffer for DDPG
class ReplayBuffer_DDPG:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32), torch.tensor(rewards, dtype=torch.float32), torch.tensor(next_states, dtype=torch.float32), torch.tensor(dones, dtype=torch.float32)

    def __len__(self):
        return len(self.buffer)