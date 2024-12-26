"""
This module simulates the environment but does not perform rendering.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import json
import numpy as np
from random import uniform, choice, randint
from math import atan, pi, sin, cos


def load(level_filename: str) -> tuple:
    """Load en environment from a file.

    Args:
        match_filename: Configuration file.
        level_filename: Level layout.

    Returns: Tuple organized as `tilemap, objects, agents`.
    """
    with open(level_filename) as f:
        content = json.load(f)
    tilemap = [list(line) for line in content]
    return tilemap


def generate_level(
        n: int,
        points: int | tuple[int] = (1, 3),
        walls: int | tuple[int] = (4, 8),
        water: int | tuple[int] = (0, 5),
        trees: int | tuple[int] = (0, 3),
        doors: int | tuple[int] = (0, 1),
        enemies: int | tuple[int] = (0, 1),
        danger: int | tuple[int] = (0, 2)
        ) -> tuple:
    """Create a random environment.

    Args:
        n: Dimension of the level.
        points: Number of points in the level. If a tuple is provided, it is
            interpreted as a range and the number of points is randomly
            sampled from it (including extrema).
        water: Amount of water tiles.
        trees: Amount of trees.
        doors: Amount of doors.
        enemies: Number of enemies.
        danger: Number of danger zones.
    """
    if type(points) == int:
        points = (points, points)
    if type(walls) == int:
        walls = (walls, walls)
    if type(water) == int:
        water = (water, water)
    if type(trees) == int:
        trees = (trees, trees)
    if type(doors) == int:
        doors = (doors, doors)
    if type(enemies) == int:
        enemies = (enemies, enemies)
    if type(danger) == int:
        danger = (danger, danger)
    tilemap = [[list("3" + " " * (n - 2) + "3") for _ in range(n)] for _ in range(n)][0]
    tilemap[0] = list("3" * n)
    tilemap[-1] = list("3" * n)
    tilemap[randint(1, n - 2)][randint(1, n - 2)] = "p"
    n_points = randint(points[0], points[1])
    while True:
        i, j = randint(1, n - 2), randint(1, n - 2)
        c = tilemap[i][j]
        if c == " ":
            tilemap[i][j] = "*"
            n_points -= 1
            if n_points <= 0:
                break

    def add_element(c: str, a: int, b: int):
        for _ in range(randint(a, b)):
            i, j = randint(1, n - 2), randint(1, n - 2)
            o = tilemap[i][j]
            if o == " ":
                tilemap[i][j] = c

    add_element("d", doors[0], doors[1])
    add_element("k", doors[0], doors[1])
    add_element("3", walls[0], walls[1])
    add_element("w", water[0], water[1])
    add_element("t", trees[0], trees[1])
    add_element("e", enemies[0], enemies[1])
    add_element("s", danger[0], danger[1])
    return tilemap


class Vector():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        return (self.x ** 2 + self.y ** 2)**0.5

    def normalize(self):
        n = self.norm()
        if n:
            self.x = self.x / n
            self.y = self.y / n

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, n):
        return Vector(self.x * n, self.y * n)

    def copy(self):
        return Vector(self.x, self.y)

    def __repr__(self):
        return f"<{self.x}, {self.y}>"

    def __eq__(self, other):
        return other.x == self.x and other.y == self.y

    def rotate(self, r):
        r = r * pi / 180
        x, y = self.x, self.y
        self.x = cos(r) * x - sin(r) * y
        self.y = sin(r) * x + cos(r) * y


class Agent:
    def __init__(self, position):
        self.position = Vector(position[0], position[1])
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

    def get_rotation(self):
        d = self.front.x if self.front.x != 0.0 else 0.01
        r = -1.0 * atan(self.front.y / d)
        if self.front.x < 0.0:
            r += pi
        return r * 180.0 / pi - 90.0


class PlayerAgent(Agent):
    def __init__(self, properties):
        Agent.__init__(self, properties)
        self.MAGIC_SPEED = 0.1
        self.speed = 6.0
        self.score = 0.0
        self.n_keys = 0

    def fire(self):
        if self.magic > 0.0:
            self.action = {"action": "fire", "direction": self.direction}
            self.magic -= 0.4


class EnemyAgent(Agent):
    def __init__(self, properties):
        Agent.__init__(self, properties)
        self.MAGIC_SPEED = 0.05
        self.speed = 2.0
        self.direction_change_period = 1.5
        self.shooting_timer = uniform(0.2, 2.0)
        self.timer = 0.0
        self.change_direction()
        self.last_position = self.position.copy()

    def change_direction(self):
        self.direction = choice(
            (
                Vector(1.0, 0.0),
                Vector(0.0, 1.0),
                Vector(-1.0, 0.0),
                Vector(0.0, -1.0)
            )
        )

    def update(self, delta: float):
        self.timer += delta
        self.shooting_timer -= delta
        if self.timer > self.direction_change_period:
            self.timer = 0.0
            self.change_direction()
        if self.position == self.last_position:
            self.change_direction()
        if self.shooting_timer < 0.0:
            self.fire()
            self.shooting_timer = uniform(0.25, 2.5)
        Agent.update(self, delta)
        self.last_position = self.position.copy()

    def fire(self):
        if self.magic > 0.0:
            self.action = {"action": "fire2", "direction": self.direction}
            self.magic -= 0.1


class Projectile:
    def __init__(self, name, position, direction):
        self.name = name
        self.position = position
        self.direction = direction
        self.position -= (self.direction * 0.25)
        self.speed = 1.0
        self.exploded = False
        self.lifetime = 3.0
        if self.name == "fire":
            self.speed = 20.0
        if self.name == "fire2":
            self.speed = 10.0

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


class Animation:
    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.total_lifetime = 0.5
        self.lifetime = 0.0


TILES = (" ", "1", "2", "3", "w")
OBJECTS = ("v", "h", "*", "k", "t", "d", "s")


class Environment:
    def __init__(
            self,
            tilemap: list[list[int]],
            ):
        self.tilemap = [
            list(c if c in TILES else " " for c in l) for l in tilemap
        ]
        self.objects = [
            list(c if c in OBJECTS else " " for c in l) for l in tilemap
        ]
        self.projectiles = []
        self.agents = []
        self.animations = []
        for i in range(len(tilemap)):
            for j in range(len(tilemap[0])):
                if tilemap[i][j] == "p":
                    self.agents.append(("player", PlayerAgent((j, i))))
                    self.player = self.agents[-1][1]
                    self.tilemap[i][j] = " "
                elif tilemap[i][j] == "e":
                    self.agents.append(("enemy", EnemyAgent((j, i))))
                    self.tilemap[i][j] = " "
        self.n_points = sum(
            [len(list(c for c in l if c == "*")) for l in tilemap]
        )
        self.n_total_points = self.n_points
        self.collisions = np.zeros((len(self.tilemap), len(self.tilemap[0])))
        self.refresh_collisions()

    def get_player(self) -> Agent:
        return self.player

    def refresh_collisions(self):
        for i in range(len(self.tilemap)):
            for j in range(len(self.tilemap[0])):
                # Normal tiles
                if self.tilemap[i][j] in (' ', '0', '1'):
                    if self.objects[i][j] in ("t", "d"):
                        self.collisions[i][j] = 1
                    else:
                        self.collisions[i][j] = 0
                elif self.tilemap[i][j] == 2:
                    self.collisions[i][j] = 1
                else:
                    self.collisions[i][j] = 2
                # Water
                if self.objects[i][j] == "w":
                    self.collisions[i][j] = 1
                # Bridges
                if self.objects[i][j] in ("v", "h"):
                    self.collisions[i][j] = 0
                    self.tilemap[i][j] = "w"

    def update(self, delta: float) -> bool:
        original_hp = self.player.health_points
        original_score = self.player.score
        self.displace_agents(delta)
        self.update_agents(delta)
        self.move_projectiles(delta)
        self.update_animations(delta)
        self.collect_objects(delta)
        self.update_objects(delta)
        final_hp = self.player.health_points
        final_score = self.player.score
        reward = (final_score - original_score) + (final_hp - original_hp)
        return reward, self.is_end_state()

    def get_score(self) -> tuple[int]:
        if self.player.health_points < 0:
            return None
        else:
            return int(self.player.score), self.n_total_points

    def is_end_state(self):
        if self.player.health_points <= 0.0:
            return True
        if self.n_points == 0:
            return True
        return False

    def update_objects(self, delta: float):
        x = int(self.player.position.x + 0.5)
        y = int(self.player.position.y + 0.5)
        for i in range(len(self.objects)):
            for j in range(len(self.objects[0])):
                if self.objects[i][j] == "d" and self.get_player().n_keys:
                    distance = Vector(x - j, y - i).norm()
                    if distance < 1.02:
                        self.get_player().n_keys -= 1
                        self.objects[i][j] = " "
                        self.collisions[i][j] = 0

    def collect_objects(self, delta: float):
        x = int(self.player.position.x + 0.5)
        y = int(self.player.position.y + 0.5)
        if x < 0:
            x = 0
        if y < -1:
            y = 0
        if x >= len(self.objects[0]):
            x = len(self.objects[0]) - 1
        if y >= len(self.objects):
            y = len(self.objects) - 1
        if self.objects[y][x] == "*":
            self.player.score += 1.0
            self.n_points -= 1
            self.objects[y][x] = " "
        elif self.objects[y][x] == "k":
            self.get_player().n_keys += 1
            self.objects[y][x] = " "
        elif self.objects[y][x] == "s":
            self.player.health_points -= delta * 0.4

    def update_animations(self, delta: float):
        retained = []
        for i in range(len(self.animations)):
            self.animations[i].lifetime += delta
            if self.animations[i].lifetime < self.animations[i].total_lifetime:
                retained.append(self.animations[i])
        self.animations = retained

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
                # Do not test collisions if there is no obstacle.
                if self.collisions[y + i][x + j] == 0:
                    continue
                # Do not push out if there is not overlap.
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
        for _, agent in self.agents:
            agent.position += agent.direction * delta * agent.speed
            self.push_out(agent)

    def move_projectiles(self, delta):
        for p in self.projectiles:
            p.update(delta)
            # Tilemap
            row, col = int(p.position.y), int(p.position.x)
            if row >= 0 and row < len(self.collisions) and col >= 0 and col < len(self.collisions[0]):
                if self.collisions[row][col] >= 2:
                    p.explode()
                if self.objects[row][col] == "t":
                    p.explode()
                    self.objects[row][col] = " "
                    self.collisions[row][col] = 0
            # Agents
            for name, agent in self.agents:
                if (
                    abs(agent.position.x - p.position.x) < 0.5
                    and abs(agent.position.y - p.position.y) < 0.5
                ):
                    p.explode()
        retained = []
        for i in range(len(self.projectiles)):
            if not self.projectiles[i].exploded:
                retained.append(self.projectiles[i])
            else:
                p = self.projectiles[i].position
                if self.projectiles[i].name == "fire":
                    self.animations.append(Animation("flame", p))
                    self.burn(p, "fire")
                elif self.projectiles[i].name == "fire2":
                    self.animations.append(Animation("flame2", p))
                    self.burn(p, "fire2")
        self.projectiles = retained

    def update_agents(self, delta):
        for name, agent in self.agents:
            agent.update(delta)
            if agent.action:
                start = agent.position + agent.front
                if agent.action["action"] == "fire":
                    self.projectiles.append(
                        Projectile("fire", start, agent.front.copy())
                    )
                if agent.action["action"] == "fire2":
                    self.projectiles.append(
                        Projectile("fire2", start, agent.front.copy())
                    )
                agent.action = None
            if agent != self.player:
                distance = (agent.position - self.player.position).norm()
                if distance < 1.0:
                    self.player.health_points -= delta * 0.2

    def burn(self, position, explosion: str):
        retained = []
        for name, agent in self.agents:
            if (
                abs(agent.position.x - position.x) < 1
                and abs(agent.position.y - position.y) < 1
            ):
                if name == "player":
                    if explosion == "fire":
                        agent.health_points -= 0.05
                    else:
                        agent.health_points -= 0.4
                else:
                    if explosion == "fire":
                        agent.health_points -= 0.51
            if agent.health_points > 0.0:
                retained.append((name, agent, ))
        self.agents = retained
