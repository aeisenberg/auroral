"""
This module simulates the environment but does not perform rendering.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import numpy as np
from random import uniform, choice, randint, random
from math import atan, pi, sin, cos


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
        self.direction = Vector(0.0, 0.0)
        self.position = position
        self.speed = 3.0

    def update(self, delta: float):
        self.position += self.direction * self.speed * delta


class PlayerAgent(Agent):
    def __init__(self):
        Agent.__init__(self, Vector(0.5, 0.9))
        self.speed = 0.75
        self.health_points = 1.0
        self.power = 1.0
        self.action = ""

    def update(self, delta: float):
        Agent.update(self, delta)
        if self.position.x < 0.00:
            self.position.x = 0.00
        if self.position.x > 0.95:
            self.position.x = 0.95
        if self.position.y < 0.1:
            self.position.y = 0.1
        if self.position.y > 0.9:
            self.position.y = 0.9
        if self.power > 0.0:
            self.power += delta * 0.2
        else:
            self.power += delta * 0.1
        if self.power > 1.0:
            self.power = 1.0

    def fire(self):
        if self.power > 0.0:
            self.action = "fire"
            self.power -= 0.2


class Projectile:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction
        self.exploded = False
        self.lifetime = 2.0

    def update(self, delta):
        self.position += self.direction * 2.0 * delta
        self.lifetime -= delta
        if self.lifetime < 0.0:
            self.exploded = True

    def explode(self):
        self.exploded = True


class Animation:
    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.total_lifetime = 0.5
        self.lifetime = 0.0


class Environment:
    def __init__(self):
        self.N_MAX_COINS = 3
        self.SCROLL_SPEED = 0.5
        self.player = PlayerAgent()
        self.score = 0
        self.projectiles = []
        self.animations = []
        self.coins = []
        self.dangers = []

    def get_player(self) -> Agent:
        return self.player

    def update(self, delta: float) -> tuple:
        original_hp = self.player.health_points
        original_power = self.player.power
        original_score = self.score
        self.displace_agents(delta)
        self.update_agents(delta)
        n_explosions = self.move_projectiles(delta)
        self.update_animations(delta)
        self.update_coins(delta)
        self.update_danger(delta)
        final_hp = self.player.health_points
        final_power = self.player.power
        final_score = self.score
        reward = 0.0
        if final_score > original_score:
            reward += 1.0
        if final_power < original_power:
            reward -= 0.1
        if final_hp < original_hp:
            reward -= 0.2
        if n_explosions:
            reward += 0.5
        lost = self.player.health_points <= 0.0
        return reward, self.is_end_state(), lost

    def get_score(self) -> tuple[int]:
        if self.player.health_points < 0:
            return None
        else:
            return int(self.player.score), self.n_total_points

    def is_end_state(self):
        if self.player.health_points <= 0.0:
            return True
        return False

    def update_coins(self, delta: float):
        if len(self.coins) < self.N_MAX_COINS and random() < delta * 0.4:
            self.coins.append(Vector(uniform(0.0, 0.9), -0.1))
        retained = []
        for coin in self.coins:
            coin.y += delta * 0.3
            d = (coin - self.player.position).norm()
            if d < 0.05:
                self.score += 1
            elif coin.y > 1.05:
                pass
            else:
                retained.append(coin)
        self.coins = retained

    def update_danger(self, delta: float):
        if len(self.dangers) < 2 and random() < delta * 0.2:
            h = uniform(0.2, 0.4)
            if len(self.dangers) == 0:
                self.dangers.append(
                    [
                        Vector(uniform(0.0, 0.9), -0.1 - h),
                        Vector(uniform(0.2, 0.4), h)
                    ]
                )
            else:
                for _ in range(5):
                    x = uniform(0.0, 0.9)
                    w = uniform(0.2, 0.4)
                    if x > self.dangers[0][0].x and x + w < self.dangers[0][0].x + self.dangers[0][1].x:
                        continue
                    else:
                        self.dangers.append(
                            [
                                Vector(x, -0.1 - h),
                                Vector(w, h)
                            ]
                        )
                        break
        retained = []
        for p, s in self.dangers:
            p.y += delta * self.SCROLL_SPEED * 0.275
            q = self.player.position
            if q.x < p.x or q.x > p.x + s.x or q.y < p.y or q.y > p.y + s.y:
                pass
            else:
                self.player.health_points -= delta * 0.2
            if p.y < 1.05:
                retained.append([p, s])
        self.dangers = retained

    def update_animations(self, delta: float):
        retained = []
        for i in range(len(self.animations)):
            self.animations[i].lifetime += delta
            if self.animations[i].lifetime < self.animations[i].total_lifetime:
                retained.append(self.animations[i])
        self.animations = retained

    def displace_agents(self, delta):
        self.player.update(delta)

    def move_projectiles(self, delta):
        explosions = 0
        for p in self.projectiles:
            p.update(delta)
        retained = []
        for i in range(len(self.projectiles)):
            if not self.projectiles[i].exploded:
                retained.append(self.projectiles[i])
            else:
                pass
        self.projectiles = retained
        return explosions

    def update_agents(self, delta):
        if self.player.action:
            start = self.player.position.copy()
            if self.player.action == "fire":
                self.projectiles.append(
                    Projectile(start, Vector(0.0, -1.0))
                )
                self.player.action = ""
