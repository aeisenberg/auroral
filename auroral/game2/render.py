"""
Render the environment on a surface.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import pygame

from auroral.game1 import environment

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


def load_resources(directory: str) -> dict:
    pygame.font.init()
    return {
        "background": pygame.image.load(directory + 'background.png'),
        "ship": pygame.image.load(directory + 'ship.png'),
        "projectiles": pygame.image.load(directory + 'projectiles.png'),
        "danger": pygame.image.load(directory + 'danger.png'),
        "coin": pygame.image.load(directory + 'coin.png'),
        "animations": pygame.image.load(directory + 'animations.png'),
        "font": pygame.font.SysFont('Liberation Mono', 24)
    }


def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


period = 0.0


def isometric(
        env: environment.Environment,
        screen,
        resources,
        dimension,
        camera: list = [0, 0],
        delta: float = 0.0
    ):
    global period
    period += delta
    W = screen.get_width()
    H = screen.get_height()
    # Background
    BACKGROUND_H = resources["background"].get_height()
    SCROLL_TIME =  5.0 / env.SCROLL_SPEED
    scroll_factor = (period % SCROLL_TIME) / SCROLL_TIME
    initial_y = H - BACKGROUND_H
    final_y = BACKGROUND_H
    offset = initial_y + (scroll_factor * final_y)
    screen.blit(
        resources["background"],
        (0, offset - BACKGROUND_H),
    )
    screen.blit(
        resources["background"],
        (0, offset),
    )
    # Projectiles
    for projectile in env.projectiles:
        p = projectile.position.y + projectile.position.x - 0.75
        ix, iy = 1, 0
        iy += int(period * 20 % 2)
        P = 2
        screen.blit(
            resources["projectiles"],
            (projectile.position.x * W, projectile.position.y * H),
            (ix + (ix * 32) + (P / 2), iy + (iy * 32) + (P / 2), 32 - P, 32 - P)
        )
    # Coins
    for coin in env.coins:
        screen.blit(resources["coin"], (coin.x * W, coin.y * H))
    # Agent
    p = env.get_player().position
    screen.blit(
        resources["ship"],
        (p.x * W, p.y * H),
        (1, 1, 32, 32)
    )
    for e in env.enemies:
        p = e.position
        screen.blit(
            resources["ship"],
            (p.x * W, p.y * H),
            (1, 34, 32, 32)
        )
    # Danger zones
    for p, s in env.dangers:
        screen.blit(
            resources["danger"],
            (p.x * W, p.y * H),
            (0, 0, s.x * W, s.y * H)
        )
    # Animations
    for a in env.animations:
        p = a.position
        iy = int(a.lifetime / a.total_lifetime * 7)
        screen.blit(
            resources["animations"],
            (p.x * W, p.y * H),
            (67, iy * 32 + iy + 1, 32, 32)
        )


def agent_state(env, screen, resources):
    pygame.draw.rect(screen, BLACK, (6, 6, 128, 16))
    pygame.draw.rect(screen, BLUE, (8, 8, env.get_player().power * 124, 12))
    pygame.draw.rect(screen, BLACK, (6, 24, 128, 16))
    pygame.draw.rect(screen, RED, (8, 26, env.get_player().health_points * 124, 12))
    text_surface = resources["font"].render(str(env.score), False, (200, 200, 255))
    screen.blit(text_surface, (6, 44))


def debug(
        screen,
        delta,
        delta_buffer,
        font
    ):
    if delta == 0.0:
        fps = "N/A"
    else:
        fps = f"{(1.0 / delta):.6}"
    text_surface = font.render("FPS: " + fps, False, (20, 175, 20))
    screen.blit(text_surface, (6, 48))
    avg_delta = sum(delta_buffer) / len(delta_buffer)
    if avg_delta == 0.0:
        fps = "N/A"
    else:
        fps = f"{(1.0 / avg_delta):.6}"
    text_surface = font.render("Avg. FPS: " + fps, False, (20, 175, 20))
    screen.blit(text_surface, (6, 70))
