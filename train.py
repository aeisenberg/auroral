"""Train reinforcement learning models.

Usage:

$ python3 train.py <configuration file>
$ python3 train.py <configuration file> --slow # Train slowly for debugging

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: December 2024
    - License: MIT
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import argparse
from time import sleep, time
import json
from random import choices
import numpy as np
import torch
from torchvision import transforms

from auroral import environment, game, render
from models.random import Random
from models import dqn

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Configure main parameters
parser = argparse.ArgumentParser(
    prog="train.py",
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

SCREEN_DIMENSIONS = (256, 256)
SLOW = args.slow
DEBUG = args.debug
NO_GRAPHICS = args.no_graphics
OUTPUT = args.output

if DEBUG:
    pygame.font.init()
    font = pygame.font.SysFont('Liberation Mono', 24)

with open(args.configuration, "r") as f:
    configuration = json.load(f)

# Create the images. `screen` is used by the model. `meta_screen` is used for
# debugging.
screen = pygame.Surface(SCREEN_DIMENSIONS)
if not NO_GRAPHICS:
    pygame.init()
    pygame.display.set_caption("Auroral - Training")
    if DEBUG:
        meta_screen = pygame.display.set_mode((700, 512))
    else:
        meta_screen = pygame.display.set_mode(SCREEN_DIMENSIONS)

MATCHES_FILE = "assets/matches.json"

resources = {}
for level_configuration in configuration["levels"]:
    theme = level_configuration["theme"]
    resources[theme] = render.load_resources("assets/", MATCHES_FILE, theme)

DELTA = 1.0 / configuration["framerate"]


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a raw RGB game frame to the format expected by the network."""
    n = configuration["input_size"]
    T = torch.Tensor(frame).permute(2, 0, 1)
    T = T.unsqueeze(0)
    T = torch.nn.functional.interpolate(T, size=(n, n), mode='bilinear')
    if configuration["n_channels"] == 1:
        T = transforms.functional.rgb_to_grayscale(T)
    T = T.squeeze(0)
    T = torch.Tensor(T).permute(2, 1, 0)
    T /= 255.0
    return T.numpy()


# Create the model.
DEVICE = configuration["device"]
N_FRAMES = configuration["n_frames"]
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

def display_info(text, line = None, x = None, y = None):
    text_surface = font.render(text, False, (255, 255, 255))
    if line is not None:
        meta_screen.blit(text_surface, (300, 24 + (24 * line)))
    else:
        meta_screen.blit(text_surface, (x, y))


# Training loop.
N_EPISODES = configuration["n_episodes"]
N_STEPS = configuration["maximum_n_steps"]
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
    env = environment.Environment(level)

    update_screen()
    if DEBUG:
        meta_screen.fill((0, 0, 0))
        cumulative_reward = 0.0

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
        state = prepare_frame(pygame.surfarray.array3d(screen))
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
            if DEBUG:
                cumulative_reward += reward
                meta_screen.fill((0, 0, 0))
                meta_screen.blit(screen, (32, 32))
                display_info(f"Game footage", x=32, y=0)
                display_info(f"DQN Input", x=32, y=320)
                display_info(f"Episode: {episode + 1}    Step: {step + 1}", line=0)
                display_info(f"Delta: {delta:.4} s", line=1)
                display_info(f"Reward: {reward:.4}", line=2)
                display_info(f"Cumulative reward: {cumulative_reward:.4}", line=3)
                display_info(f"ϵ = {epsilon:.4}", line=4)
                model_input = (state * 255.0).astype(int)
                if configuration["n_channels"] == 1:
                    model_input = model_input.squeeze()
                model_input = pygame.surfarray.make_surface(model_input)
                meta_screen.blit(model_input, (32, 352))
            else:
                meta_screen.blit(screen, (0, 0))
            pygame.display.update()
            sleep(10)

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
