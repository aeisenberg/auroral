"""
Main loop of the game.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import os
import pygame
import time

import environment
import renderer


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


ENVIRONMENT_FILE = "../assets/levels/test.json"
MATCHES_FILE = "../assets/matches.json"
TILE_SIZE = 32
SCREEN_DIMENSIONS = (TILE_SIZE * 16, TILE_SIZE * 16)


tilemap, objects, agents, theme = environment.load(ENVIRONMENT_FILE)
env = environment.Environment(tilemap, objects, agents)
player = env.get_player()
resources = renderer.load_resources("../assets/", MATCHES_FILE, theme)
pygame.init()
pygame.display.set_caption("Auroral")
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)

ti = time.time()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_a):
                player.direction.x = -1.0
            if event.key in (pygame.K_RIGHT, pygame.K_d):
                player.direction.x = 1.0
            if event.key in (pygame.K_UP, pygame.K_w):
                player.direction.y = -1.0
            if event.key in (pygame.K_DOWN, pygame.K_s):
                player.direction.y = 1.0
            if event.key in (pygame.K_p, ):
                player.fire()
        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_LEFT, pygame.K_a) and player.direction.x < 0:
                player.direction.x = 0
            if event.key in (pygame.K_RIGHT, pygame.K_d) and player.direction.x > 0:
                player.direction.x -= 1.0
            if event.key in (pygame.K_UP, pygame.K_w) and player.direction.y < 0:
                player.direction.y = 0
            if event.key in (pygame.K_DOWN, pygame.K_s) and player.direction.y > 0:
                player.direction.y = 0
    now = time.time()
    delta = now - ti
    ti = now
    env.update(delta)
    position = env.get_player().position
    renderer.render(
        env,
        screen,
        resources,
        SCREEN_DIMENSIONS,
        (position.x, position.y),
        TILE_SIZE
    )
    pygame.display.update()
    if delta > 0.0:
        print(f"FPS: {1.0 / delta}")
