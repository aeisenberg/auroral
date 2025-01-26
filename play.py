"""
Play the game. Intended to be used by humans for testing. Or for fun!

Usage:

>>> python3 play.py
>>> python3 play.py --debug # Add debugging information
>>> python3 play.py -s X # Change the size of the window to X by X pixels

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import argparse
from time import sleep

from auroral.game1 import game as game1
from auroral.game2 import game as game2

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(
    prog="Auroral",
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
parser.add_argument(
    "-g",
    "--game",
    help="Game to play. `1` or `2`. Default: 2.",
    type=int
)
parser.add_argument(
    "-l",
    "--level",
    help="Level to use for game 1.",
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
pygame.display.set_caption("Auroral - Game")
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)

ALL_LEVELS = [f for f in os.listdir("assets/levels") if f.endswith(".json")]
ALL_THEMES = [f for f in os.listdir("assets/themes") if f.isnumeric()]

if args.game == 1:
    game1.play(screen, args.level)
elif args.game in (None, 2):
    game2.play(screen)
