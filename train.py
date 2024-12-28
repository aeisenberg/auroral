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
from random import uniform, choices, choice

from auroral import environment, game, render
from models.random import Random
from models import dqn

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
        meta_screen = pygame.display.set_mode((650, 256))
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
    model = Random(10, 25)
elif configuration["model"] == "dqn-1-shallow":
    model = dqn.DQN_1_shallow()
    model.to("cuda")
elif configuration["model"] == "dqn-1-mid":
    model = dqn.DQN(dqn.DQN_1_mid(), dqn.DQN_1_mid())


try:
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_parameters}")
except:
    pass



# Training loop.
N_EPISODES = configuration["n_episodes"]
INITIAL_EPSILON = 0.99
training_start_time = time()

for episode in range(N_EPISODES):
    print(f"Episode {episode}")
    epsilon = (1.0 - (episode / N_EPISODES) ) * INITIAL_EPSILON
    episode_start_time = time()
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
        meta_screen.blit(text_surface, (280, 24 + (24 * line)))

    # Prepare the next episode.
    update_screen()
    if DEBUG:
        meta_screen.fill((0, 0, 0))
        cumulative_reward = 0.0

    N_STEPS = configuration["maximum_n_steps"]
    pure_exploitation = (DEBUG and (episode + 1) % 10 == 0)
    if pure_exploitation:
        model.prepare_episode(0.0)
    else:
        model.prepare_episode(epsilon)
    for step in range(N_STEPS):
        t0 = time()
        if not pure_exploitation:
            print(f"\033[FEpisode: {episode + 1} / {N_EPISODES}. "
                + f"Step: {step + 1} / {N_STEPS}"
                + f"    Episode duration (s): {(t0 - episode_start_time):.4}"
                + f"    Training duration (s): {(t0 - training_start_time):.4}"
            )
        state = pygame.surfarray.array3d(screen).copy()
        action = model.act(state)
        if pure_exploitation:
            print(f"{model.prediction(state)}")
        reward, done = game.frame(env, DELTA, action)

        update_screen()
        next_state = pygame.surfarray.array3d(screen).copy()
        if not pure_exploitation:
            model.step(state / 255.0, action, reward, next_state / 255.0, done)
        t1 = time()
        delta = t1 - t0
        if not NO_GRAPHICS:
            meta_screen.blit(screen, (0, 0))
            pygame.display.update()
            if DEBUG:
                cumulative_reward += reward
                meta_screen.fill((0, 0, 0))
                display_info(f"Episode: {episode + 1}    Step: {step + 1}", 0)
                display_info(f"Delta: {delta:.4} s", 1)
                display_info(f"Reward: {reward:.4}", 2)
                display_info(f"Cumulative reward: {cumulative_reward:.4}", 3)
                if pure_exploitation:
                    display_info(f"Explore: 0.0. Exploit: 1.0", 4)
                else:
                    display_info(f"Explore: {epsilon:.2}. Exploit: {(1.0 - epsilon):.2}", 4)

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
