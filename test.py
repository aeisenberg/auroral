"""Test a trained model."""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import argparse
from time import sleep, time
import json
from random import choices
from collections import deque
import datetime
import numpy as np
import torch
from torchvision import transforms

from auroral import environment, game, models, render


def configure() -> dict:
    """Read the command line arguments and configuration file.

    Returns: Dictionary containing all configuration values.
    """
    parser = argparse.ArgumentParser(
        prog="test.py",
        description="Test a trained model."
    )
    parser.add_argument(
        "configuration",
        help="Filepath to the model.",
        type=str
    )
    args = parser.parse_args()

    with open(args.configuration + "/configuration.json", "r") as f:
        configuration = json.load(f)

    if configuration["model"] == "dqn-1-shallow":
        network = models.DQN_1_shallow
    elif configuration["model"] == "dqn-1-mid":
        network = models.DQN_1_mid
    model = models.DQN(
        network,
        configuration["device"],
        configuration["frame_size"],
        configuration["n_frames"],
        configuration["n_channels"],
        configuration["learning_rate"],
        configuration["batch_size"],
        configuration["target_update_frequency"],
    )
    model.load(args.configuration + "/model.pt")
    return configuration, model


def load_resources(configuration: dict) -> dict:
    """Load images required to display the game.

    Args:
        configuration: Configuration values.
    """
    MATCHES_FILE = "assets/matches.json"
    resources = {}
    for level_configuration in configuration["levels"]:
        theme = level_configuration["theme"]
        resources[theme] = render.load_resources(
            "assets/",
            MATCHES_FILE,
            theme
        )
    return resources


def prepare_game(configuration: dict) -> tuple:
    """Prepare the Pygame window and surface.

    Args:
        configuration: Configuration values.

    Returns: Tuple containing:
        - `screen`: The pygame surface onto which the game is displayed.
        - `meta_screen`: Display screen used for debugging (`None` if unused).
        - `font`: Font used to print debugging information in the window.
    """
    font, meta_screen = None, None
    screen = pygame.display.set_mode((256, 256))
    pygame.init()
    pygame.display.set_caption("Auroral - Test")
    return screen, meta_screen, font


def update_screen(
        env: environment.Environment,
        screen: pygame.Surface,
        theme: dict,
        configuration: dict
    ) -> None:
    """Update the game screen used by the agent."""
    screen.fill((0, 0, 0))
    position = env.get_player().position
    render.isometric(
        env,
        screen,
        theme,
        (256, 256),
        (position.x, position.y),
        1.0 / configuration["framerate"]
    )


def prepare_frame(frame: np.ndarray, configuration: dict) -> np.ndarray:
    """Convert a raw RGB game frame to the format expected by the network."""
    n = configuration["frame_size"]
    T = torch.Tensor(frame).permute(2, 0, 1)
    T = T.unsqueeze(0)
    T = torch.nn.functional.interpolate(T, size=(n, n), mode='bilinear')
    if configuration["n_channels"] == 1:
        T = transforms.functional.rgb_to_grayscale(T)
    T = T.squeeze(0)
    T = torch.Tensor(T).permute(2, 1, 0)
    T = torch.Tensor(T).permute(2, 1, 0)
    T /= 255.0
    return T.numpy()


def observe(
        env: environment.Environment,
        screen: pygame.Surface,
        configuration: dict,
        theme: dict,
        buffer: deque
    ) -> torch.Tensor:
    """Observe the state of the environment."""
    update_screen(env, screen, theme, configuration)
    array = pygame.surfarray.array3d(screen)
    array = prepare_frame(array, configuration)
    tensor =  torch.Tensor(array)
    buffer.append(tensor)
    return torch.cat(tuple(buffer), dim=0)


def create_buffer(
        env: environment.Environment,
        screen: pygame.Surface,
        configuration: dict,
        theme: dict,
    ) -> deque:
    """Initialize a frame buffer to store states."""
    buffer = deque(maxlen=configuration["n_frames"])
    for _ in range(configuration["n_frames"]):
        observe(env, screen, configuration, theme, buffer)
    return buffer


configuration, model = configure()
screen, meta_screen, font = prepare_game(configuration)
resources = load_resources(configuration)
DELTA = 1.0 / configuration["framerate"]

quit = False
outcomes = []
for level in range(1, 11):
    tilemap = environment.load(f"assets/levels/{level}.json")
    env = environment.Environment(tilemap)
    theme = resources[configuration["evaluation_level"]["theme"]]
    buffer = create_buffer(env, screen, configuration, theme)
    cumulative_reward = 0.0
    episode_start_time = time()
    for step in range(500):
        t0 = time()
        state = observe(env, screen, configuration, theme, buffer)
        action = model.act(state, 0.05)
        reward, done, lost = game.frame(env, DELTA, action)
        next_state = observe(env, screen, configuration, theme, buffer)
        model.step(state, action, reward, next_state, done)
        t1 = time()
        cumulative_reward += reward
        delta = t1 - t0
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
        if done or quit:
            break
        # if delta < DELTA:
        #    sleep(DELTA - delta)
    if done and not lost:
        outcomes.append("s")
    else:
        outcomes.append("f")
    if quit == True:
        break

print(outcomes)
