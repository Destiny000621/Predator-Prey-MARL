from collections import namedtuple, deque
import random
import numpy as np
import torch

# Define a named tuple to represent a single experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.stack([torch.tensor(e.state, dtype=torch.float32) for e in experiences])
        actions = torch.stack([torch.tensor(e.action, dtype=torch.float32) for e in experiences])
        rewards = torch.stack([torch.tensor(e.reward, dtype=torch.float32) for e in experiences])
        next_states = torch.stack([torch.tensor(e.next_state, dtype=torch.float32) for e in experiences])
        dones = torch.stack([torch.tensor(e.done, dtype=torch.float32) for e in experiences])
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# Test the ReplayBuffer class
buffer_size = 10000
batch_size = 64
replay_buffer = ReplayBuffer(buffer_size, batch_size)
OBS_DIM = 16
ACT_DIM = 5

# Adding dummy data to test sampling
for _ in range(100):
    state = np.random.randn(OBS_DIM)
    action = np.random.choice(ACT_DIM)
    reward = np.random.randn()
    next_state = np.random.randn(OBS_DIM)
    done = np.random.choice([True, False])
    replay_buffer.add(state, action, reward, next_state, done)

sampled_data = replay_buffer.sample()
sampled_data[0].shape, sampled_data[1].shape, sampled_data[2].shape, sampled_data[3].shape, sampled_data[4].shape