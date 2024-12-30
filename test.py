"""A self-contained test for a DQN.

The goal of the reinforcement learning task is for the agent (a red square) to
reach a target (a green square). The environment is a 1D or 2D matrix.
Configure the section labelled "PARAMETERS" to modify the test.

Usage:

$ python3 test.py

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: December 2024
    - License: MIT
"""

import numpy as np
import pygame
from time import sleep
from random import randint, random, sample
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# PARAMETERS
DEVICE = "cuda"  # Use "cpu" if CUDA is not available.
GAME_SIZE = 8  # Length of the environment.
GAME_DIMENSIONS = 2  # Dimension of the environment, either 1 or 2.
N_FRAMES= 4  # Number of game frames fed to the DQN.
LEARNING_RATE = 1e-3  # Learning rate of the Adam optimizer
USE_CONVOLUTIONAL_LAYER = True  # Apply a convolutional layer to the input of the DQN
USE_MEMORY_REPLAY = True  # Use a memory replay buffer.

N_EPISODES = 1000  # Total number of episodes during training
SCREEN_DIMENSION = (350, 256)  # Dimension of the screen used for visualization
N_STEPS = GAME_SIZE * GAME_DIMENSIONS * 2  # Maximum number of steps in an episode
BATCH_SIZE = 32  # Number of data points in a batch
TARGET_UPDATE_FREQUENCY = 1000  # Update frequency of the target DQN in number of steps
EVALUATION_FREQUENCY = 1000  # Evaluation frequency in number of steps
EVALUATION_N_EPISODES = 100  # Number of episodes to test the DQN
MEMORY_SIZE = 10000  # Size of the memory replay buffer
GAMMA = 0.99  # Discount factor
REWARDS = {  # Rewards given to the DQN for each event in the simulation
    "reach objective": 10,
    "move closer to objective": 1,
    "move away from objective": -1,
    "remain static": -0.5,
    "out of bound penalty": -1
}
N_ACTIONS = 4  # Total number of actions


# DNQ
class DQN(nn.Module):
    def __init__(self, n):
        super(DQN, self).__init__()
        n2 = int(n / 2)
        # 1D
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.network = nn.Sequential(
            nn.Linear(n, n2),
            nn.ReLU(),
            nn.Linear(n2, N_ACTIONS)
        )
        # 1D Stacked
        self.network_stacked = nn.Sequential(
            nn.Linear(n * N_FRAMES, n),
            nn.ReLU(),
            nn.Linear(n, n2),
            nn.ReLU(),
            nn.Linear(n2, N_ACTIONS)
        )
        # 2D
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.network2d = nn.Sequential(
            nn.Flatten(),
            nn.Linear(GAME_SIZE ** 2, GAME_SIZE * 2),
            nn.ReLU(),
            nn.Linear(GAME_SIZE * 2, GAME_SIZE),
            nn.ReLU(),
            nn.Linear(GAME_SIZE, N_ACTIONS)
        )
        self.network2d_stacked = nn.Sequential(
            nn.Flatten(),
            nn.Linear(GAME_SIZE ** 2 * N_FRAMES, GAME_SIZE ** 2),
            nn.ReLU(),
            nn.Linear(GAME_SIZE ** 2, GAME_SIZE),
            nn.ReLU(),
            nn.Linear(GAME_SIZE, N_ACTIONS)
        )
        self.output = nn.Softmax(dim=0)

    def forward(self, x):
        if GAME_DIMENSIONS == 1:
            if USE_CONVOLUTIONAL_LAYER:
                if N_FRAMES == 1:
                    x = x.unsqueeze(1)
                    x = self.conv1d(x)
                    x = torch.flatten(x, start_dim=1)
                    return torch.sigmoid(self.network(x))
                else:
                    x = x.view(x.size(0), -1)
                    x = x.unsqueeze(1)
                    x = self.conv1d(x)
                    x = torch.flatten(x, start_dim=1)
                    return torch.sigmoid(self.network_stacked(x))
            else:
                if N_FRAMES == 1:
                    return torch.sigmoid(self.network(x))
                else:
                    x = x.view(x.size(0), -1)
                    return torch.sigmoid(self.network_stacked(x))
        else:
            if USE_CONVOLUTIONAL_LAYER:
                if N_FRAMES == 1:
                    exit()
                else:
                    x = x.unsqueeze(2)
                    batch_size, num_frames, _, _, _ = x.shape
                    x = x.view(batch_size * num_frames, 1, x.size(3), x.size(4))
                    x = self.conv2d(x)
                    x = x.view(batch_size, num_frames, -1)
                    x = x.view(batch_size, -1)
                    return torch.sigmoid(self.network2d_stacked(x))
            else:
                if N_FRAMES == 1:
                    return torch.sigmoid(self.network2d(x))
                else:
                    x = x.view(x.size(0), -1)
                    return torch.sigmoid(self.network2d_stacked(x))


# ENVIRONMENT
screen = pygame.display.set_mode(SCREEN_DIMENSION)
pygame.init()
pygame.display.set_caption("Auroral")


class Environment:
    def __init__(self, n, dimension):
        self.n = n
        self.dimension = dimension
        self.grids = deque(maxlen=N_FRAMES)
        if dimension == 1:
            self.grid = np.zeros(n)
            self.objective = randint(0, n - 1)
            while True:
                agent_position = randint(0, n - 1)
                if agent_position != self.objective:
                    self.agent_position = agent_position
                    break
        else:
            self.grid = np.zeros((n, n))
            self.objective = [randint(0, n - 1), randint(0, n - 1)]
            while True:
                agent_position = [randint(0, n - 1), randint(0, n - 1)]
                if agent_position != self.objective:
                    self.agent_position = agent_position
                    break
        for _ in range(N_FRAMES):
            self.grids.append(self.grid.copy())

    def observe(self):
        if self.dimension == 1:
            for i in range(self.n):
                if i == self.agent_position:
                    self.grid[i] = 0.5
                elif i == self.objective:
                    self.grid[i] = 1.0
                else:
                    self.grid[i] = 0.0
        else:
            self.grid = np.zeros((self.n, self.n))
            self.grid[self.agent_position[0], self.agent_position[1]] = 0.5
            self.grid[self.objective[0], self.objective[1]] = 1.0
        if N_FRAMES == 1:
            return self.grid.copy()
        else:
            self.grids.append(self.grid.copy())
            grids = [g for g in self.grids]
            return np.stack(grids, axis=0)

    def render(self, screen):
        normal = (255, 255, 255)
        agent = (255, 0, 0)
        target = (0, 255, 0)
        s = 8
        if self.dimension == 1:
            for i in range(self.n):
                square = pygame.Rect((s + 1) * (i + 1), s, s, s)
                if i == self.objective:
                    pygame.draw.rect(screen, target, square)
                elif i == self.agent_position:
                    pygame.draw.rect(screen, agent, square)
                else:
                    pygame.draw.rect(screen, normal, square)
        else:
            for i in range(self.n):
                for j in range(self.n):
                    square = pygame.Rect((s + 1) * (i + 1), (s + 1) * (j + 1), s, s)
                    if [i, j] == self.objective:
                        pygame.draw.rect(screen, target, square)
                    elif [i, j] == self.agent_position:
                        pygame.draw.rect(screen, agent, square)
                    else:
                        pygame.draw.rect(screen, normal, square)

    def update(self, action: list[int]) -> tuple:
        oob_penalty = 0
        if self.dimension == 1:
            distance1 = abs(self.agent_position - self.objective)
            if action[0] == 1:
                self.agent_position -= 1
                if self.agent_position < 0:
                    self.agent_position = 0
                    oob_penalty = 1
            elif action[1] == 1:
                self.agent_position += 1
                if self.agent_position > self.n - 1:
                    self.agent_position = self.n - 1
                    oob_penalty = 1
            distance2 = abs(self.agent_position - self.objective)
        else:
            distance1 = abs(self.agent_position[0] - self.objective[0]) + abs(self.agent_position[1] - self.objective[1])
            if action[0] == 1:
                self.agent_position[0] -= 1
                if self.agent_position[0] < 0:
                    self.agent_position[0] = 0
                    oob_penalty = 1
            elif action[1] == 1:
                self.agent_position[0] += 1
                if self.agent_position[0] > self.n - 1:
                    self.agent_position[0] = self.n - 1
                    oob_penalty = 1
            elif action[2] == 1:
                self.agent_position[1] -= 1
                if self.agent_position[1] < 0:
                    self.agent_position[1] = 0
                    oob_penalty = 1
            elif action[3] == 1:
                self.agent_position[1] += 1
                if self.agent_position[1] > self.n - 1:
                    self.agent_position[1] = self.n - 1
                    oob_penalty = 1
            distance2 = abs(self.agent_position[0] - self.objective[0]) + abs(self.agent_position[1] - self.objective[1])
        done = self.agent_position == self.objective
        travel = distance1 - distance2
        if travel == 0:
            reward = REWARDS["remain static"]
        elif travel > 0:
            reward = REWARDS["move closer to objective"]
        else:
            reward = REWARDS["move away from objective"]
        if done:
            reward = REWARDS["reach objective"]
        reward += REWARDS["out of bound penalty"] * oob_penalty
        return reward, done


# TRAINING
policy_net = DQN(GAME_SIZE).to(DEVICE)
target_net = DQN(GAME_SIZE).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)


def select_action(state, epsilon):
    if random() < epsilon:
        action = [0 for _ in range(N_ACTIONS)]
        action[randint(0, len(action) - 1)] = 1
        return action
    else:
        state = torch.FloatTensor(np.stack(np.array([state]), axis=0)).to(DEVICE)
        Q = policy_net(state)[0].tolist()
        return [1 if q == max(Q) else 0 for q in Q]


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    batch = sample(memory, BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(np.stack(state_batch, axis=0)).to(DEVICE)
    action_batch = torch.LongTensor(np.stack(action_batch, axis=0)).to(DEVICE)
    reward_batch = torch.FloatTensor(np.stack(reward_batch, axis=0)).to(DEVICE)
    next_state_batch = torch.FloatTensor(np.stack(next_state_batch, axis=0)).to(DEVICE)
    done_batch = torch.FloatTensor(np.stack(done_batch, axis=0)).to(DEVICE)

    # Compute Q-values for current states
    q_values = policy_net(state_batch)
    q_values = action_batch * q_values
    # print("Q: " + str(q_values[0]))
    q_values = torch.sum(q_values, dim=1)

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(dim=1)[0]
        target_q_values = reward_batch + GAMMA * max_next_q_values * (1 - done_batch)

    # print("S: " + str(state_batch[0]))
    # print("A: " + str(action_batch[0]))
    # print("R: " + str(reward_batch[0]))
    # print("N: " + str(next_state_batch[0]))
    # print("D: " + str(done_batch[0]))
    # print("M: " + str(max_next_q_values[0]))
    # print("T: " + str(target_q_values[0]))
    # exit()

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate(slow_down = False):
    print("EVALUATING")
    exit_loop = False
    durations = []
    for _ in range(EVALUATION_N_EPISODES):
        env = Environment(GAME_SIZE, GAME_DIMENSIONS)
        for step in range(GAME_SIZE * 4):
            # Act
            state = env.observe()
            action = select_action(state, 0.0)
            state = torch.FloatTensor(np.stack(np.array([state]), axis=0)).to(DEVICE)
            Q = target_net(state)[0].tolist()
            action = [1 if q == max(Q) else 0 for q in Q]
            _, done = env.update(action)
            # Display
            screen.fill((0, 0, 0))
            env.render(screen)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_loop = True
            if done:
                break
            if slow_down:
                sleep(0.01)
        durations.append(step + 1)
        if exit_loop:
            break

    avg = sum(durations) / len(durations)
    print(f"Average: {avg}")
    print()
    return avg


# TRAINING LOOP
print()
steps_done = 0
averages = []
exit_loop = False
for episode in range(N_EPISODES):
    epsilon = 1.05 - (episode / N_EPISODES)
    env = Environment(GAME_SIZE, GAME_DIMENSIONS)
    for step in range(N_STEPS):
        steps_done += 1
        # Act
        state = env.observe()
        action = select_action(state, epsilon)
        reward, done = env.update(action)
        next_state = env.observe()
        memory.append((state, action, reward, next_state, done))
        # Train
        optimize_model()
        if steps_done % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if steps_done % EVALUATION_FREQUENCY == 0:
            averages.append(evaluate())
        # Display
        screen.fill((0, 0, 0))
        env.render(screen)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_loop = True
        if done or exit_loop:
            break
    print(f"\033[FEpisode {episode} finished after {step} steps. {steps_done} steps in total.")
    if exit_loop:
        break

evaluate(True)

fig, ax = plt.subplots()
ax.plot(list(range(len(averages))), averages)
ax.set(xlabel='episode', ylabel='n steps')
ax.grid()
plt.show()
pygame.quit()
