"""
Train reinforcement learning models.

Usage:

>>> python3 train.py <configuration file>
>>> python3 train.py <configuration file> --slow # Train slowly for debugging

The configuration file provided to the script must have the following content:

```
{
    "name": <Name of the model stored in the "models" directory to use>,
    "maximum_n_steps": <Maximum umber of steps in on episode>,
    "n_episodes": <Maximum number of episodes>,
    "framerate": <Number of frames per seconds (e.g. 30)>,
    "levels": [
        {
            "frequency": <Number between 0 and 1 indicating how frequently this
                level should be used for training.>,
            "theme": <Name of the level's theme>,
            "n": <Dimension of the level (int)>,
            "points": <Number of points (either int or pair of ints)>,
            "walls": <Number of walls (either int or pair of ints)>,
            "water": <Number of water tiles (either int or pair of ints)>,
            "trees": <Number of trees (either int or pair of ints)>,
            "doors": <Number of doors (either int or pair of ints)>,
            "enemies": <Number of enemies (either int or pair of ints)>,
            "danger": <Number of danger tiles (either int or pair of ints)>
        }
    ]
}
```

"""

import os
import pygame
import argparse
from time import sleep, time
import json
from random import choices
from sys import stdout

from auroral import environment, game, render
from models import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(
    prog="Auroral gym",
    description="Launch training."
)
parser.add_argument(
    "configuration",
    help="Configuration file for training.",
    type=str
)
parser.add_argument(
    "-s",
    "--slow",
    help="Train in real time to ease debugging. If not provided, run as fast as possible.",
    action="store_true"
)
parser.add_argument(
    "-d",
    "--debug",
    help="Add debugging information.",
    action="store_true"
)
parser.add_argument(
    "-n",
    "--no-graphics",
    help="Deactivate display and print information in the terminal only. Speeds up training.",
    action="store_true"
)
parser.add_argument(
    "-o",
    "--output",
    help="File in which to save the trained model.",
    type=str
)
args = parser.parse_args()

# Configure main parameters
SCREEN_DIMENSIONS = (256, 256)
SLOW = args.slow
DEBUG = args.debug
NO_GRAPHICS = args.no_graphics
OUTPUT = args.output

if DEBUG:
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', 24)

with open(args.configuration, "r") as f:
    configuration = json.load(f)

# Create the images. `screen` is used by the model. `meta_screen` is used for
# debugging.
screen = pygame.Surface(SCREEN_DIMENSIONS)
if not NO_GRAPHICS:
    pygame.init()
    pygame.display.set_caption("Auroral")
    if DEBUG:
        meta_screen = pygame.display.set_mode((600, 512))
    else:
        meta_screen = pygame.display.set_mode(SCREEN_DIMENSIONS)

MATCHES_FILE = "assets/matches.json"

resources = {}
for level_configuration in configuration["levels"]:
    theme = level_configuration["theme"]
    resources[theme] = render.load_resources("assets/", MATCHES_FILE, theme)

DELTA = 1.0 / configuration["framerate"]

# Create the model.
if configuration["model"] == "random":
    model = random.Model(10, 25)

# Training loop.
N_EPISODES = configuration["n_episodes"]
for episode in range(N_EPISODES):
    print(f"Episode {episode}")
    level_configuration = choices(
        population=configuration["levels"],
        weights=[l["frequency"] for l in configuration["levels"]],
        k=1
    )[0]
    theme = level_configuration["theme"]
    level = environment.generate_level(
        level_configuration["n"],
        level_configuration["points"],
        level_configuration["walls"],
        level_configuration["water"],
        level_configuration["trees"],
        level_configuration["doors"],
        level_configuration["enemies"],
        level_configuration["danger"],
    )
    env= environment.Environment(level)

    def update_screen():
        screen.fill((50, 50, 50))
        position = env.get_player().position
        render.isometric(
            env,
            screen,
            resources[theme],
            SCREEN_DIMENSIONS,
            (position.x, position.y),
            DELTA
        )

    def display_info(text, line):
        text_surface = font.render(text, False, (255, 255, 255))
        meta_screen.blit(text_surface, (300, 24 + (24 * line)))

    # Prepare the next episode.
    update_screen()
    model.prepare_episode()
    if DEBUG:
        meta_screen.fill((0, 0, 0))
        cumulative_reward = 0.0

    N_STEPS = configuration["maximum_n_steps"]
    for step in range(N_STEPS):
        print(f"\033[FEpisode: {episode} / {N_EPISODES}. Step: {step + 1} / {N_STEPS}")
        t0 = time()
        state = pygame.surfarray.array3d(screen).copy()
        action = model.act(state)
        reward, done = game.frame(env, DELTA, action)
        update_screen()
        next_state = pygame.surfarray.array3d(screen).copy()
        model.step(state, action, reward, next_state, done)
        t1 = time()
        delta = t1 - t0
        if not NO_GRAPHICS:
            meta_screen.blit(screen, (0, 0))
            pygame.display.update()
            if DEBUG:
                cumulative_reward += reward
                meta_screen.fill((0, 0, 0))
                display_info(f"Episode: {episode}    Step: {step}", 0)
                display_info(f"Delta: {delta:.4} s", 1)
                display_info(f"Cumulative reward: {cumulative_reward}", 2)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        if done:
            break

        if SLOW and delta < DELTA:
            sleep(DELTA - delta)

if OUTPUT:
    model.save(OUTPUT)
