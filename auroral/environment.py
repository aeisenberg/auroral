"""
This module simulates the environment but does not perform rendering.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import json
import numpy as np
from math import atan, pi


def load(level_filename: str) -> tuple:
    """Load en environment from a file.

    Args:
        match_filename: Configuration file.
        level_filename: Level layout.

    Returns: Tuple organized as `tilemap, objects, agents`.
    """
    with open(level_filename) as f:
        content = json.load(f)
    tilemap = content["tilemap"]
    objects = content["objects"]
    agents = content["agents"]
    theme = content["theme"]
    return tilemap, objects, agents, theme


class Vector():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        return (self.x ** 2 + self.y ** 2)**0.5

    def normalize(self):
        n = self.norm()
        self.x = self.x / n
        self.y = self.y / n

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, n):
        return Vector(self.x * n, self.y * n)

    def copy(self):
        return Vector(self.x, self.y)

    def __repr__(self):
        return f"<{self.x}, {self.y}>"


class Agent:
    def __init__(self, properties):
        self.position = Vector(
            properties["position"][0],
            properties["position"][1]
        )
        self.direction = Vector(0.0, 0.0)
        self.front = Vector(0.0, 1.0)  # Faces South by default.
        self.speed = 3.0
        self.s = 0.75
        self.offset = (1.0 - self.s) / 2.0
        if self.offset < 0.0:
            self.offset = 0.0
        self.health_points = 1.0
        self.magic = 1.0
        self.action = None
        self.MAGIC_SPEED = 0.0

    def update(self, delta: float):
        self.magic += delta * self.MAGIC_SPEED
        if (self.magic > 1.0):
            self.magic = 1.0
        if self.direction.norm() > 0.1:
            self.front = self.direction.copy()
            self.front.normalize()


class PlayerAgent(Agent):
    def __init__(self, properties):
        Agent.__init__(self, properties)
        self.MAGIC_SPEED = 0.05
        self.speed = 4.0

    def fire(self):
        if self.magic > 0.0:
            self.action = {"action": "fire", "direction": self.direction}
            self.magic -= 0.2

    def freeze(self):
        if self.magic > 0.0:
            self.action = {"action": "freeze", "direction": self.direction}
            self.magic -= 0.5


class Projectile:
    def __init__(self, name, position, direction):
        self.name = name
        self.position = position
        self.direction = direction
        self.speed = 1.0
        self.exploded = False
        self.lifetime = 3.0
        if self.name == "fire":
            self.speed = 15.0

    def update(self, delta):
        self.position += self.direction * self.speed * delta
        self.lifetime -= delta
        if self.lifetime < 0.0:
            self.exploded = True

    def get_rotation(self):
        d = self.direction.x if self.direction.x != 0.0 else 0.01
        r = -1.0 * atan(self.direction.y / d)
        if self.direction.x < 0.0:
            r += pi
        return r * 180.0 / pi - 90.0

    def explode(self):
        self.exploded = True


class Environment:
    def __init__(
            self,
            tilemap: list[list[int]],
            objects: list,
            agents: list
            ):
        self.tilemap = np.array(tilemap)
        self.objects = np.array(objects)
        self.projectiles = []
        self.agents = []
        for k, v in agents.items():
            if k == "player":
                self.agents.append((k, PlayerAgent(v)))
                self.player = self.agents[-1][1]
            else:
                self.agents.append((k, Agent(v)))
        self.collisions = np.zeros((len(self.tilemap), len(self.tilemap[0])))
        self.refresh_collisions()

    def get_player(self) -> Agent:
        return self.player

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
        self.update_agents(delta)
        self.move_projectiles(delta)

    def push_out(self, agent):
        x, y = int(agent.position.x), int(agent.position.y)
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
                if (agent.position.x + agent.s < x + j
                        or agent.position.x > x + j + 1
                        or agent.position.y + agent.s < y + i
                        or agent.position.y > y + i + 1):
                    continue
                # Push out.
                directions = [0.0, 0.0, 0.0, 0.0]
                if agent.position.x + agent.s > x + j:
                    directions[0] = x + j - agent.s - agent.position.x
                if agent.position.x < x + j + 1:
                    directions[1] = x + j + 1 - agent.position.x
                if agent.position.y + agent.s > y + i:
                    directions[2] = y + i - agent.s - agent.position.y
                if agent.position.y < y + i + 1:
                    directions[3] = y + i + 1 - agent.position.y
                magnitudes = [abs(d) for d in directions]
                index = min(range(len(magnitudes)), key=magnitudes.__getitem__)
                if index < 2:
                    agent.position.x += directions[index]
                else:
                    agent.position.y += directions[index]

    def displace_agents(self, delta):
        for name, agent in self.agents:
            agent.position += agent.direction * delta * agent.speed
            self.push_out(agent)

    def move_projectiles(self, delta):
        for p in self.projectiles:
            p.update(delta)
            for name, agent in self.agents:
                if (
                    agent.position.x - p.position.y < 0.5
                    and agent.position.x - p.position.y < 0.5
                ):
                    pass#p.explode()
        retained = []
        for i in range(len(self.projectiles)):
            if not self.projectiles[i].exploded:
                retained.append(self.projectiles[i])
        self.projectiles = retained

    def update_agents(self, delta):
        for name, agent in self.agents:
            agent.update(delta)
            if agent.action:
                start = agent.position + agent.front
                if agent.action["action"] == "fire":
                    self.projectiles.append(Projectile("fire", start, agent.front.copy()))
                elif agent.action["action"] == "freeze":
                    pass
                agent.action = None

