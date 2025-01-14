"""
This module simulates the environment but does not perform rendering.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

from random import uniform, choice, random
from math import pi, sin, cos
from pygame.mixer import music


class Vector():
    """A 2D vector with basic arithmetic operations."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        """Return the length of the vector."""
        return (self.x ** 2 + self.y ** 2)**0.5

    def normalize(self):
        """Change the elements of the vector to keep the orientation but have
        a norm of 1.0."""
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
        """Rotate the vector by the specified amount of radians."""
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


class EnemyAgent(Agent):
    def __init__(self, position):
        Agent.__init__(self, position)
        self.speed = 0.2
        self.direction_change_period = 1.5
        self.shooting_timer = uniform(0.2, 2.0)
        self.timer = 0.0
        self.change_direction()
        self.last_position = self.position.copy()
        self.action = ""

    def change_direction(self):
        self.direction = choice(
            (
                Vector(1.0, 0.0),
                Vector(-1.0, 0.0),
                Vector(-1.0, 1.0),
                Vector(1.0, 1.0),
                Vector(0.0, 1.0),
            )
        )

    def update(self, delta: float):
        self.timer += delta
        self.shooting_timer -= delta
        if self.timer > self.direction_change_period:
            self.timer = 0.0
            self.change_direction()
        Agent.update(self, delta)
        self.last_position = self.position.copy()


class PlayerAgent(Agent):
    def __init__(self):
        Agent.__init__(self, Vector(0.5, 0.85))
        self.speed = 0.75
        self.health_points = 1.0
        self.power = 1.0
        self.action = ""

    def update(self, delta: float):
        Agent.update(self, delta)
        if self.position.x < 0.00:
            self.position.x = 0.00
        if self.position.x > 0.9:
            self.position.x = 0.9
        if self.position.y < 0.3:
            self.position.y = 0.3
        if self.position.y > 0.85:
            self.position.y = 0.85
        if self.power > 0.0:
            self.power += delta * 0.2
        else:
            self.power += delta * 0.1
        if self.power > 1.0:
            self.power = 1.0

    def fire(self):
        if self.power > 0.0:
            self.action = "fire"
            self.power -= 0.1

    def heal(self, hp) -> None:
        self.health_points += hp
        if self.health_points > 1.0:
            self.health_points = 1.0


class Projectile:
    def __init__(self, position, direction, name, speed = 2.0):
        self.position = position
        self.direction = direction
        self.exploded = False
        self.lifetime = 2.0
        self.name = name
        self.speed = speed

    def update(self, delta):
        self.position += self.direction * self.speed * delta
        self.lifetime -= delta
        if self.lifetime < 0.0:
            self.exploded = True

    def explode(self):
        self.exploded = True


class Animation:
    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.total_lifetime = 0.25
        self.lifetime = 0.0


class Environment:
    def __init__(self, use_audio = False):
        self.N_MAX_COINS = 1
        self.N_MAX_DANGER_ZONES = 3
        self.N_MAX_ENEMIES = 3
        self.SCROLL_SPEED = 0.5
        self.player = PlayerAgent()
        self.enemies = [
            EnemyAgent(Vector(uniform(0.2, 0.8), 0.1)),
            EnemyAgent(Vector(uniform(0.2, 0.8), 0.1)),
        ]
        self.score = 0
        self.projectiles = []
        self.animations = []
        self.coins = []
        self.dangers = []
        self.audio = use_audio
        if self.audio:
            self.sounds = {
                "fire": "assets/sound/enemy.mp3",
                "damage": "assets/sound/laser.mp3",
            }

    def get_player(self) -> Agent:
        return self.player

    def update(self, delta: float) -> tuple:
        original_hp = self.player.health_points
        original_power = self.player.power
        original_score = self.score
        original_position = self.player.position.copy()
        self.displace_agents(delta)
        self.update_agents(delta)
        self.move_projectiles(delta)
        self.update_animations(delta)
        self.update_coins(delta)
        self.update_danger(delta)
        final_hp = self.player.health_points
        final_power = self.player.power
        final_score = self.score
        final_position = self.player.position.copy()
        reward = 0.0
        if final_position == original_position:
            reward -= 0.05
        if final_score >= original_score + 2:
            reward += 2.0
        elif final_score >= original_score + 1:
            reward += 1.0
        if final_power < original_power:
            reward -= 0.05
        if final_hp < original_hp:
            reward -= 0.3
        lost = self.player.health_points <= 0.0
        return reward, self.is_end_state(), lost

    def get_score(self) -> int:
        return int(self.score)

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
            if abs(coin.x - self.player.position.x) < 0.1 and 0 < self.player.position.y - coin.y < 0.1:
                self.score += 1
                self.player.heal(0.05)
            elif coin.y > 1.05:
                pass
            else:
                retained.append(coin)
        self.coins = retained

    def update_danger(self, delta: float):
        if len(self.dangers) < self.N_MAX_DANGER_ZONES and random() < delta * 0.4:
            h = uniform(0.2, 0.5)
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
            if q.x + 0.1 < p.x or q.x > p.x + s.x or q.y < p.y or q.y > p.y + s.y:
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
        for enemy in self.enemies:
            enemy.update(delta)

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
        self.update_player(delta)
        if len(self.enemies) < self.N_MAX_ENEMIES:
            if len(self.enemies) == 0 or random() < delta * 2.0:
                self.enemies.append(EnemyAgent(Vector(uniform(0.2, 0.8), -0.09)))
        retained = []
        for enemy in self.enemies:
            if random() < delta * 0.4:
                start = enemy.position.copy()
                self.projectiles.append(
                    Projectile(start, Vector(0.0, 1.0), "fire2", 1.0)
                )
            oob = enemy.position.x < -0.1 or enemy.position.x > 1.1 or enemy.position.y < -0.1 or enemy.position.y > 1.1
            collided = False
            if (enemy.position - self.player.position).norm() < 0.05:
                self.player.health_points -= delta * 0.6
            for p in self.projectiles:
                if p.name == "fire":
                    if abs(p.position.x - enemy.position.x - 0.025) > 0.1:
                        continue
                    if p.position.y > enemy.position.y:
                        continue
                    if abs(p.position.y - enemy.position.y) > 0.25:
                        continue
                    collided = True
                    p.explode()
                    self.animations.append(Animation("ascii", enemy.position))
                    self.score += 2
            if oob or collided:
                pass
            else:
                retained.append(enemy)
        self.enemies = retained

    def update_player(self, delta):
        if self.player.action:
            start = self.player.position.copy()
            if self.player.action == "fire":
                self.projectiles.append(
                    Projectile(start, Vector(0.0, -1.0), "fire", 6.0)
                )
                self.player.action = ""
                if self.audio:
                    music.load(self.sounds["fire"])
                    music.play()
        retained = []
        for p in self.projectiles:
            if p.name == "fire2" and (p.position - self.player.position).norm() < 0.05:
                p.explode()
                self.animations.append(Animation("ascii2", p.position))
                self.player.health_points -= 0.1
                if self.audio:
                    music.load(self.sounds["damage"])
                    music.play()
            else:
                retained.append(p)
        self.projectiles = retained
