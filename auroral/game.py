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


ENVIRONMENT_FILE = "../assets/test_configuration.json"
TILE_SIZE = 32
SCREEN_DIMENSIONS = (TILE_SIZE * 16, TILE_SIZE * 16)


tilemap, objects, agents = environment.load(ENVIRONMENT_FILE)
env = environment.Environment(tilemap, objects, agents)
player = env.get_player()
resources = renderer.load_resources("../assets/", ENVIRONMENT_FILE)
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
                player.direction[0] = -1.0
            if event.key in (pygame.K_RIGHT, pygame.K_d):
                player.direction[0] = 1.0
            if event.key in (pygame.K_UP, pygame.K_w):
                player.direction[1] = -1.0
            if event.key in (pygame.K_DOWN, pygame.K_s):
                player.direction[1] = 1.0
        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_LEFT, pygame.K_a) and player.direction[0] < 0:
                player.direction[0] = 0
            if event.key in (pygame.K_RIGHT, pygame.K_d) and player.direction[0] > 0:
                player.direction[0] -= 1.0
            if event.key in (pygame.K_UP, pygame.K_w) and player.direction[1] < 0:
                player.direction[1] = 0
            if event.key in (pygame.K_DOWN, pygame.K_s) and player.direction[1] > 0:
                player.direction[1] = 0
    now = time.time()
    delta = now - ti
    ti = now
    env.update(delta)
    renderer.render(env, screen, resources, SCREEN_DIMENSIONS, env.get_player().position)
    pygame.display.update()
