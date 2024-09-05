"""
This module simulates the environment but does not perform rendering.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import json
import numpy as np


def load(filename: str) -> tuple:
    """Load en environment from a file.

    Args:
        filename: Configuration file.

    Returns: Tuple organized as `tilemap, objects, agents`.
    """
    with open(filename) as f:
        content = json.load(f)
    tilemap = content["tilemap"]
    objects = content["objects"]
    agents = content["agents"]
    return tilemap, objects, agents


class Agent:
    def __init__(self, properties):
        self.position = properties["position"]
        self.direction = [0.0, 0.0]
        self.speed = properties["speed"]
        self.s = properties["dimension"]
        self.offset = (1.0 - self.s) / 2.0
        if self.offset < 0.0:
            self.offset = 0.0


class Environment:
    def __init__(
            self,
            tilemap: list[list[int]],
            objects: list,
            agents: list
            ):
        self.tilemap = np.array(tilemap)
        self.objects = np.array(objects)
        self.agents = []
        for k, v in agents.items():
            self.agents.append((k, Agent(v)))
        self.collisions = np.zeros((len(self.tilemap), len(self.tilemap[0])))
        self.refresh_collisions()

    def get_player(self) -> Agent:
        for k, v in self.agents:
            if k == "player":
                return v

    def refresh_collisions(self):
        for i in range(len(self.tilemap)):
            for j in range(len(self.tilemap[0])):
                # Normal tiles
                if self.tilemap[i][j] in ('0', '1'):
                    self.collisions[i][j] = 0
                else:
                    self.collisions[i][j] = 1
                # Bridges
                if self.tilemap[i][j] == "2":
                    if self.objects[i][j] in ("1", "2"):
                        self.collisions[i][j] = 0
                    else:
                        self.collisions[i][j] = 1

    def update(self, delta: float):
        self.displace_agents(delta)

    def push_out(self, agent):
        x, y = int(agent.position[0]), int(agent.position[1])
        if x < 0:
            x = 0
        if y < -1:
            y = 0
        if x >= len(self.collisions[0]):
            x = len(self.collisions[0]) - 1
        if y >= len(self.collisions):
            y = len(self.collisions) - 1
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if y + i >= len(self.collisions) or x + j >= len(self.collisions[0]):
                    continue
                # There is can obstacle.
                if self.collisions[y + i][x + j] == 0:
                    continue
                # There is a collision.
                if (agent.position[0] + agent.s < x + j
                        or agent.position[0] > x + j + 1
                        or agent.position[1] + agent.s < y + i
                        or agent.position[1] > y + i + 1):
                    continue
                # Push out.
                directions = [0.0, 0.0, 0.0, 0.0]
                if agent.position[0] + agent.s > x + j:
                    directions[0] = x + j - agent.s - agent.position[0]
                if agent.position[0] < x + j + 1:
                    directions[1] = x + j + 1 - agent.position[0]
                if agent.position[1] + agent.s > y + i:
                    directions[2] = y + i - agent.s - agent.position[1]
                if agent.position[1] < y + i + 1:
                    directions[3] = y + i + 1 - agent.position[1]
                magnitudes = [abs(d) for d in directions]
                index = min(range(len(magnitudes)), key=magnitudes.__getitem__)
                if index < 2:
                    agent.position[0] += directions[index]
                else:
                    agent.position[1] += directions[index]

    def displace_agents(self, delta):
        for name, agent in self.agents:
            agent.position[0] += agent.direction[0] * delta * agent.speed
            agent.position[1] += agent.direction[1] * delta * agent.speed
            self.push_out(agent)
