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
import argparse
from collections import deque

import environment
from environment import Vector
import renderer


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", help="Add debug information.",
                    action="store_true")
args = parser.parse_args()
DEBUG = args.debug
if DEBUG:
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', 24)
    delta_buffer = deque(maxlen=500)

ENVIRONMENT_FILE = "../assets/levels/isometric_test.json"
MATCHES_FILE = "../assets/matches.json"
SCREEN_DIMENSIONS = (512, 512)


tilemap, objects, agents, theme = environment.load(ENVIRONMENT_FILE)
env = environment.Environment(tilemap, objects, agents)
player = env.get_player()
resources = renderer.load_resources("../assets/", MATCHES_FILE, theme)
pygame.init()
pygame.display.set_caption("Auroral")
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)

ti = time.time()

direction = Vector(0.0, 0.0)

while True:
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
    env.update(delta)
    position = env.get_player().position
    screen.fill((50, 50, 50))
    renderer.render_isometric(
        env,
        screen,
        resources,
        SCREEN_DIMENSIONS,
        (position.x, position.y),
        delta
    )
    renderer.render_agent_state(env, screen)
    if DEBUG:
        delta_buffer.append(delta)
        renderer.render_debug(
            screen,
            delta,
            delta_buffer,
            font
        )
    pygame.display.update()
