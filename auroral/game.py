"""
Main loop of the game.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import pygame
import time
from collections import deque

from auroral.environment import Vector
from auroral import environment, render


MATCHES_FILE = "assets/matches.json"


def play(
        screen: pygame.Surface,
        level_file: str,
        theme: str,
        debug: bool
    ) -> tuple[int]:
    """Play the game in interactive mode.

    This function is intended to be used as a game loop for a human player.

    Args:
        screen: The Pygame surface ono which to render the game.
        level_file: File path used to load the level. `None` if random.
        theme: The graphic theme to use.
        debug: If `True`, display debugging information on the screen.
    """
    dimensions = (screen.get_width(), screen.get_height())
    try:
        tilemap = environment.load(level_file)
    except:
        tilemap = environment.generate_level(16)
    env = environment.Environment(tilemap)
    player = env.get_player()
    resources = render.load_resources("assets/", MATCHES_FILE, theme)
    ti = time.time()
    direction = Vector(0.0, 0.0)
    if debug:
        pygame.font.init()
        font = pygame.font.SysFont('Liberation Mono', 24)
        delta_buffer = deque(maxlen=500)

    while True:
        # Inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    direction.x -= 1.0
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    direction.x += 1.0
                if event.key in (pygame.K_UP, pygame.K_w):
                    direction.y -= 1.0
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    direction.y += 1.0
                if event.key in (pygame.K_p, ):
                    player.fire()
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    direction.x += 1.0
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    direction.x -= 1.0
                if event.key in (pygame.K_UP, pygame.K_w):
                    direction.y += 1.0
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    direction.y -= 1.0
            player.direction = direction.copy()
            player.direction.normalize()
            player.direction.rotate(-45)
        now = time.time()
        delta = now - ti
        ti = now
        # Update the game.
        _, is_over, _ = env.update(delta)
        if is_over:
            return env.get_score()
        # Render on the surface.
        screen.fill((0, 0, 0))
        position = env.get_player().position
        render.isometric(
            env,
            screen,
            resources,
            dimensions,
            (position.x, position.y),
            delta
        )
        render.agent_state(env, screen)
        if debug:
            delta_buffer.append(delta)
            render.debug(
                screen,
                delta,
                delta_buffer,
                font
            )
        pygame.display.update()


def frame(
        env: environment.Environment,
        delta: float,
        action: dict
    ) -> tuple:
    """Process one frame of the game logic and display the environment.

    This function is intended to be used in non-interactive mode to train
    machine learning models.

    Args:
        env: Game environment.
        screen: pygame surface onto which to render the game (observation).
        resources: Game assets used to render the environment.
        delta: Number of seconds between two frames.
        action: Dictionary of inputs. The keys are `up`, `down`, `left`,
            `right`, and `fire`. The values are Booleans.

    Returns: A tuple organized as (reward: float, is_terminal_state: bool),
        where `reward` indicates the reward collected by the agent at the end
        of the frame (after completing its action) and `is_terminal_state` is
        `True` is the game is completed and `False` if not.
    """
    player = env.get_player()
    direction = Vector(0.0, 0.0)
    if len(action) < 5:
        action = {
            "up": action[0],
            "down": action[1],
            "left": action[2],
            "right": action[3]
        }
    else:
        action = {
            "up": action[0],
            "down": action[1],
            "left": action[2],
            "right": action[3],
            "fire": action[4],
        }
    if action["up"]:
        direction.y -= 1.0
    if action["down"]:
        direction.y += 1.0
    if action["left"]:
        direction.x -= 1.0
    if action["right"]:
        direction.x += 1.0
    if "fire" in action and action["fire"]:
        player.fire()
    player.direction = direction.copy()
    player.direction.normalize()
    player.direction.rotate(-45)
    reward, done, lost = env.update(delta)
    return reward, done, lost
