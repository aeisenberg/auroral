from collections import deque
from random import randint, sample, random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize


N_ACTIONS = 5


class DQN():

    def __init__(
            self,
            network: nn.Module,
            device: str,
            frame_size: int,
            n_frames: int,
            n_channels: int,
            learning_rate: float,
            batch_size: int,
            target_update_frequency: int
        ):
        self.device = device
        self.policy_net = network(frame_size, n_frames, n_channels).to(self.device)
        self.target_net = network(frame_size, n_frames, n_channels).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = deque(maxlen=1000)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        self.batch_size = batch_size
        self.gamma = 0.99
        self.step_count = 0
        self.target_update_frequency = target_update_frequency

    def act(self, state: np.ndarray, epsilon: float):
        """
        Produces an action based on the game frame.
        Args:
            state: The game state as a (256, 256, 3) RGB image.
            epsilon: Probability of choosing a random action.
        Returns:
            np.ndarray: A binary vector of size 5 indicating the chosen actions.
        """
        if random() < epsilon:
            action = [0 for _ in range(N_ACTIONS)]
            action[randint(0, len(action) - 1)] = 1
            return action
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state).tolist()[0]
                return [1 if q == max(q_values) else 0 for q in q_values]

    def save(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath, weights_only=True))

    def step(
            self,
            state: np.ndarray,
            action: list[int],
            reward: float,
            next_state: np.ndarray,
            done: bool) -> None:
        """
        Performs a single training step.

        Args:
            state (np.ndarray): Current state oof the environment (1 frame).
            action (list): Binary vector of the action taken.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Next state (1 frame)
            done (bool): Whether the episode is finished.
        """
        self.policy_net.train()
        self.step_count += 1
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return

        # Sample from memory
        batch = sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.FloatTensor(np.stack(state_batch, axis=0)).to(self.device)
        action_batch = torch.LongTensor(np.stack(action_batch, axis=0)).to(self.device)
        reward_batch = torch.FloatTensor(np.stack(reward_batch, axis=0)).to(self.device)
        next_state_batch = torch.FloatTensor(np.stack(next_state_batch, axis=0)).to(self.device)
        done_batch = torch.FloatTensor(np.stack(done_batch, axis=0)).to(self.device)

        # Compute Q-values for current states
        q_values = self.policy_net(state_batch)
        q_values = action_batch * q_values
        q_values = torch.sum(q_values, dim=1)

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(dim=1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.step_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class DQN_1_shallow(nn.Module):
    """Single-frame deep Q network.

    Expected shape of the input: [batch_size, n_channels * n_frames, n, n],
    where n is the size of the image.
    """

    def __init__(self, frame_size: int, n_frames: int, n_channels: int):
        """
        Args:
            frame_size: Dimension of the input image.
            n_frames: Number of images in the input.
            n_channels: Number of channels in the input - either 1 or 3.
        """
        super(DQN_1_shallow, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=n_frames * n_channels,
            out_channels=32,  # Number of filters in the first layer
            kernel_size=8,    # Size of the convolutional kernel
            stride=4,         # Stride for downsampling
            padding=0         # No padding
        )
        output_size = int((frame_size - 8) / 4 + 1)
        self.fc1 = nn.Linear(32 * output_size * output_size, 512)
        self.fc2 = nn.Linear(512, N_ACTIONS)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class DQN_1_mid(nn.Module):
    """Deep Q network."""

    def __init__(self, frame_size: int, n_frames: int):
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

    def forward(self, x):
        return torch.sigmoid(self.network(x))
