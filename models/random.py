"""Baseline model. Simply picks a random action.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: December 2024
    - License: MIT
"""

import json
from random import choice
import numpy as np


class Random():

    def __init__(self, direction_change_frequency: int, fire_frequency: int):
        """Basic model.

        Args:
            direction_change_frequency: The number of actions required to
                change the direction of the agent.
            fire_frequency: The number of actions required to fire a
                projectile.
        """
        self._direction_change_frequency = direction_change_frequency
        self._change_direction = direction_change_frequency
        self._fire_frequency = fire_frequency
        self._fire = fire_frequency
        self._current_direction = self._generate_direction()

    def _generate_direction(self):
        return choice(
            (
                (0.0,   1.0),
                (0.0,  -1.0),
                (1.0,  -1.0),
                (1.0,   0.0),
                (1.0,   1.0),
                (-1.0, -1.0),
                (-1.0,  0.0),
                (-1.0,  1.0),
            )
        )

    def prepare_episode(self) -> None:
        pass

    def act(self, state: np.ndarray) -> dict:
        self._change_direction -= 1
        if self._change_direction <= 0:
            self._change_direction = self._direction_change_frequency
            self._current_direction = self._generate_direction()
        action = {
            "up": 1 if self._current_direction[1] < 0 else 0,
            "down": 1 if self._current_direction[1] > 0 else 0,
            "left": 1 if self._current_direction[0] < 0 else 0,
            "right": 1 if self._current_direction[0] > 0 else 0,
            "fire": 0
        }
        self._fire -= 1
        if self._fire <= 0:
            action["fire"] = 1
            self._fire = self._fire_frequency
        return action

    def step(
            self,
            state: np.ndarray,
            action: dict,
            reward: float,
            next_state: np.ndarray,
            done: bool) -> None:
        pass

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            content = {
                "direction": self._direction_change_frequency,
                "fire": self._fire_frequency
            }
            json.dump(content, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            content = json.load(f)
            self._direction_change_frequency = content["direction"]
            self._fire_frequency = content["fire"]
