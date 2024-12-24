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
from auroral import environment, renderer


MATCHES_FILE = "assets/matches.json"


def play(
        screen: pygame.Surface,
        level_file: str,
        theme: str,
        debug: bool
    ) -> tuple[int]:
    dimensions = (screen.get_width(), screen.get_height())
    tilemap = environment.load(level_file)
    env = environment.Environment(tilemap)
    player = env.get_player()
    resources = renderer.load_resources("assets/", MATCHES_FILE, theme)
    ti = time.time()
    direction = Vector(0.0, 0.0)
    if debug:
        pygame.font.init()
        font = pygame.font.SysFont('Comic Sans MS', 24)
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
        if env.update(delta):
            return env.get_score()
        # Render on the surface.
        screen.fill((50, 50, 50))
        position = env.get_player().position
        renderer.render_isometric(
            env,
            screen,
            resources,
            dimensions,
            (position.x, position.y),
            delta
        )
        renderer.render_agent_state(env, screen)
        if debug:
            delta_buffer.append(delta)
            renderer.render_debug(
                screen,
                delta,
                delta_buffer,
                font
            )
        pygame.display.update()
