from collections import namedtuple, deque
import random
import numpy as np
import torch

#Replay Buffer for MADDPG
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    # Define a named tuple to represent a single experience
    Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.Experience(state, action, reward, next_state, done)
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
