"""
Utility script.

Usage:
    py play.py  # Start the game in interactive mode.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import os
import pygame
import argparse
import pygame

from auroral import game

ENVIRONMENT_FILE = "assets/levels/test/isometric.json"
SCREEN_DIMENSIONS = (512, 512)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(
    prog='Auroal',
    description='Launch the game'
)
parser.add_argument(
    "-d",
    "--debug",
    help="Add debug information.",
    action="store_true"
)
args = parser.parse_args()
DEBUG = args.debug

pygame.init()
pygame.display.set_caption("Auroral")
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)

ALL_LEVELS = [f for f in os.listdir("assets/levels") if f.endswith(".json")]


def menu(level: int) -> int:
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', 24)
    stop = False
    while not stop:
        screen.fill((50, 50, 50))
        text_surface = font.render(f"LEVEL", False, (255, 255, 255))
        screen.blit(
            text_surface,
            (
                SCREEN_DIMENSIONS[0] / 2 - text_surface.get_width() / 2,
                SCREEN_DIMENSIONS[1] / 3)
        )
        text_surface = font.render(f"<- {level} ->", False, (255, 255, 255))
        screen.blit(
            text_surface,
            (
                SCREEN_DIMENSIONS[0] / 2 - text_surface.get_width() / 2,
                SCREEN_DIMENSIONS[1] / 2
            )
        )
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    level -= 1
                    if level < 1:
                        level = len(ALL_LEVELS)
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    level += 1
                    if level > len(ALL_LEVELS):
                        level = 1
                if event.key in (pygame.K_RETURN, ):
                    stop = True
                if event.key in (pygame.K_ESCAPE, ):
                    level = None
                    stop = True
    return level


level = 1


while True:
    level = menu(level)
    if level is None:
        break
    file = f"assets/levels/{ALL_LEVELS[level - 1]}"
    game.play(screen, file, SCREEN_DIMENSIONS, DEBUG)
