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
from time import sleep

from auroral import game

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(
    prog="Auroal",
    description="Launch the game."
)
parser.add_argument(
    "-d",
    "--debug",
    help="Add debug information.",
    action="store_true"
)
parser.add_argument(
    "-s",
    "--size",
    help="Define the size of the screen. Default: 512",
    type=int
)

args = parser.parse_args()
DEBUG = args.debug
s = args.size
if s == None:
    s = 512
if s < 256:
    print("Error: the screen size must be equal or greater than 256.")
    exit()
SCREEN_DIMENSIONS = (s, s) if s else (256, 256)

pygame.init()
pygame.display.set_caption("Auroral")
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)

ALL_LEVELS = [f for f in os.listdir("assets/levels") if f.endswith(".json")]
ALL_THEMES = [f for f in os.listdir("assets/themes") if f.isnumeric()]


def menu(level: int, theme: int) -> int:
    pygame.font.init()
    focus = 0
    instruction_font = pygame.font.SysFont('Comic Sans MS', 14)
    font = pygame.font.SysFont('Comic Sans MS', 24)
    stop = False

    def render(f, text, y):
        text_surface = f.render(text, False, (255, 255, 255))
        screen.blit(
            text_surface,
            (SCREEN_DIMENSIONS[0] / 2 - text_surface.get_width() / 2, y)
        )

    while not stop:
        screen.fill((50, 50, 50))
        text = f"Move with the arrows or AWSD."
        render(instruction_font, text, 10)
        text = f"Begin the level with ENTER."
        render(instruction_font, text, 40)
        text = f"-> LEVEL: {level}" if focus == 0 else f"LEVEL: {level}"
        if level == 0:
            text = f"-> LEVEL: Random" if focus == 0 else f"LEVEL: Random"
        render(font, text, SCREEN_DIMENSIONS[1] / 3)
        text = f"-> THEME: {theme}" if focus == 1 else f"THEME: {theme}"
        render(font, text, SCREEN_DIMENSIONS[1] / 2)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    if focus == 0:
                        level -= 1
                        if level < 0:
                            level = len(ALL_LEVELS)
                    elif focus == 1:
                        theme -= 1
                        if theme < 0:
                            theme = len(ALL_THEMES)
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    if focus == 0:
                        level += 1
                        if level > len(ALL_LEVELS):
                            level = 0
                    elif focus == 1:
                        theme += 1
                        if theme > len(ALL_THEMES):
                            theme = 1
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    focus += 1
                    if focus > 1:
                        focus = 0
                if event.key in (pygame.K_UP, pygame.K_w):
                    focus -= 1
                    if focus < 0:
                        focus = 1
                if event.key in (pygame.K_RETURN, ):
                    stop = True
                if event.key in (pygame.K_ESCAPE, ):
                    level = None
                    stop = True
    return level, theme


def display_score(screen, score: tuple[int]) -> None:
    font = pygame.font.SysFont('Comic Sans MS', 24)
    w, h = screen.get_width(), screen.get_height()
    if score is None:
        screen.fill((155, 0, 0))
        text_surface = font.render("X_X", False, (255, 255, 255))
        screen.blit(
            text_surface,
            (w / 2 - text_surface.get_width() / 2, h / 2)
        )
    else:
        screen.fill((0, 155, 0))
        text_surface = font.render(
            f"{score[0]} / {score[1]}", False, (255, 255, 255)
        )
        screen.blit(
            text_surface,
            (w / 2 - text_surface.get_width() / 2, h / 2)
        )
    pygame.display.update()
    sleep(1)


level = 1
theme = 1


while True:
    level, theme = menu(level, theme)
    if level is None:
        break
    file = f"assets/levels/{level}.json" if level > 0 else ""
    score = game.play(screen, file, str(theme), DEBUG)
    display_score(screen, score)
