import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
HIDDEN_SIZE = 128
LR = 0.001

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, state_size, action_size, seed, HIDDEN_SIZE, learning_rate=LR):
        self.actor = Actor(state_size, action_size, seed, HIDDEN_SIZE).to(device)
        self.critic = Critic(state_size, seed, HIDDEN_SIZE).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.actor(state).cpu()
        # Sample an action from the output probabilities
        action = torch.multinomial(probs, 1).item()
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        action = torch.tensor(action).view(-1, 1).to(device)
        reward = torch.tensor(reward).float().view(-1, 1).to(device)
        done = torch.tensor(done).float().view(-1, 1).to(device)

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
        policy_loss = -torch.log(probs.gather(1, action)) * advantage.detach()
        self.optimizer_actor.zero_grad()
        policy_loss.mean().backward()
        self.optimizer_actor.step()
