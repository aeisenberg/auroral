from collections import deque
from random import randint, sample, random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN():

    def __init__(self, dqn, target_dqn):
        self.device = "cuda"
        self.policy_net = dqn.to(self.device)
        self.target_net = target_dqn.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(1000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-8)
        self.epsilon = 1.0
        self.batch_size = 32
        self.gamma = 0.99
        self.step_count = 0
        self.target_update_frequency = 1000

    def prepare_episode(self, epsilon) -> None:
        self.epsilon = epsilon

    def act(self, state: np.ndarray):
        """
        Produces an action based on the game frame.
        Args:
            state (np.ndarray): The game state as a (256, 256, 3) RGB image.
        Returns:
            np.ndarray: A binary vector of size 5 indicating the chosen actions.
        """
        if random() < self.epsilon:
            return [randint(0, 1) for _ in range(5)]
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state).tolist()[0]
                return [1 if q > 0.5 else 0 for q in q_values]
                # return [1 if q == max(q_values) else 0 for q in q_values]

    def prediction(self, state: np.ndarray):
        """
        Produces an action based on the game frame.
        Args:
            state (np.ndarray): The game state as a (256, 256, 3) RGB image.
        Returns:
            np.ndarray: A binary vector of size 5 indicating the chosen actions.
        """
        self.policy_net.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.float().squeeze(0).tolist()

    def save(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath, weights_only=True))

    def step(
            self,
            state: np.ndarray,
            action: dict,
            reward: float,
            next_state: np.ndarray,
            done: bool) -> None:
        """
        Performs a single training step.
        Args:
            state (np.ndarray): Current state as a (256, 256, 3) RGB image.
            action (dict): Dictionary of actions taken, mapping indices to binary values.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Next state as a (256, 256, 3) RGB image.
            done (bool): Whether the episode is finished.
        """
        self.policy_net.train()
        self.step_count += 1
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) < self.batch_size:
            return

        # Sample from replay memory
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        states = torch.stack([torch.FloatTensor(s).permute(2, 0, 1) for s in batch[0]]).to(self.device)
        actions = torch.FloatTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.stack([torch.FloatTensor(s).permute(2, 0, 1) for s in batch[0]]).to(self.device)
        dones = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # Compute Q values and targets
        q_values = self.policy_net(states) * actions
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, targets * actions)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.step_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class DQN_1_shallow(nn.Module):
    """Single-frame deep Q network."""

    def __init__(self):
        super(DQN_1_shallow, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # Output: [32, 63, 63]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [32, 31, 31]
        self.fc1 = nn.Linear(32 * 31 * 31, 512)  # Flatten and connect to 512 neurons
        self.fc2 = nn.Linear(512, 5)  # Output layer: 5 Q-values for the 5 actions
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary outputs

        self.optimizer = optim.Adam(self.parameters(), lr=1e-8)
        self.loss_fn = nn.CrossEntropyLoss()

        self.GAMMA = 0.99

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # Batch size, Flattened features
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class DQN_1_mid(nn.Module):
    """Deep Q network."""

    def __init__(self):
        super(DQN_1_mid, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # Adjust based on input size
            nn.ReLU(),
            nn.Linear(512, 5),
        )
        self.resizer = Resize((84, 84))

    def forward(self, x):
        x = self.resizer(x)
        return torch.sigmoid(self.network(x))
