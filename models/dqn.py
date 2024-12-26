from random import uniform, randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BasicDQN():
    """Base class for a deep Q network."""

    def __init__(self):
        self.experience_buffer = []
        self.sampling_probability = 0.5
        self.buffer_len = 100

    def prepare_episode(self) -> None:
        self.train()

    def act(self, state: np.ndarray):
        """
        Produces an action based on the game frame.
        Args:
            state (np.ndarray): The game state as a (256, 256, 3) RGB image.
        Returns:
            np.ndarray: A binary vector of size 5 indicating the chosen actions.
        """
        self.eval()
        with torch.no_grad():
            frame_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to("cuda").float()
            q_values = self(frame_tensor)[0]
        return {
            "up": 0 if q_values[0] < 0.5 else 1,
            "down": 0 if q_values[1] < 0.5 else 1,
            "left": 0 if q_values[2] < 0.5 else 1,
            "right": 0 if q_values[3] < 0.5 else 1,
            "fire": 0 if q_values[4] < 0.5 else 1,
        }

    def prediction(self, state: np.ndarray):
        """
        Produces an action based on the game frame.
        Args:
            state (np.ndarray): The game state as a (256, 256, 3) RGB image.
        Returns:
            np.ndarray: A binary vector of size 5 indicating the chosen actions.
        """
        self.eval()
        with torch.no_grad():
            frame_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to("cuda").float()
            q_values = self(frame_tensor)[0]
        return [float(i) for i in q_values.cpu().numpy()]

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
        if uniform(0.0, 1.0) < self.sampling_probability:
            self.experience_buffer.append((state.copy(), action, reward, next_state.copy(), done))

        if len(self.experience_buffer) >= self.buffer_len - 1:
            i = randint(0, len(self.experience_buffer) - 1)
            args = self.experience_buffer.pop(i)
            self._step(*args)

    def _step(self,
            state: np.ndarray,
            action: dict,
            reward: float,
            next_state: np.ndarray,
            done: bool) -> None:
        self.train()
        # Convert states to tensors
        state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to("cuda").float()
        next_state_tensor = torch.from_numpy(next_state).permute(2, 0, 1).unsqueeze(0).to("cuda").float()

        # Compute Q-values for the current state and the next state
        q_values = self(state_tensor).squeeze(0)
        next_q_values = self(next_state_tensor).squeeze(0)

        # Extract the Q-values corresponding to the taken actions
        action_tensor = torch.tensor(list(action.values())).to("cuda")

        # Compute target Q-values using the Bellman equation
        with torch.no_grad():
            target_q_values = reward + (self.GAMMA * next_q_values * (1 - int(done)))

        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values * action_tensor, target_q_values * action_tensor)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        self.optimizer.step()

        # Compute loss
        #loss = self.loss_fn(q_values * action_tensor, target_q_values * action_tensor)


class DQN_1_shallow(nn.Module, BasicDQN):
    """Single-frame deep Q network."""

    def __init__(self):
        super(DQN_1_shallow, self).__init__()
        BasicDQN.__init__(self)
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


class DQN_1_mid(nn.Module, BasicDQN):
    """Deep Q network."""

    def __init__(self):
        super(DQN_1_mid, self).__init__()
        BasicDQN.__init__(self)
        # Convolutional layers without max pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # Output: (32, 62, 62)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # Output: (64, 30, 30)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1) # Output: (128, 28, 28)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 5)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-8)
        self.loss_fn = nn.MSELoss()

        self.GAMMA = 0.99

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))  # Output values between 0 and 1
        return x
