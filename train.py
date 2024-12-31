"""Train reinforcement learning models.

Usage:

$ python3 train.py <configuration file>
$ python3 train.py <configuration file> --debug  # Display debugging info.
$ python3 train.py <configuration file> --slow  # Train slowly for debugging

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

from auroral import environment, game, models, render

# Change the work directory to retrieve the asset files reliably.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def configure() -> dict:
    """Read the command line arguments and configuration file.

    Returns: Dictionary containing all configuration values.
    """
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

    with open(args.configuration, "r") as f:
        configuration = json.load(f)

    configuration["screen_dimension"] = (256, 256)
    configuration["slow"] = args.slow
    configuration["debug"] = args.debug
    configuration["no_graphics"] = args.no_graphics
    configuration["output"] = args.output
    return configuration


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
    if configuration["debug"]:
        pygame.font.init()
        font = pygame.font.SysFont('Liberation Mono', 24)

    screen = pygame.Surface(configuration["screen_dimension"])
    if not configuration["no_graphics"]:
        pygame.init()
        pygame.display.set_caption("Auroral - Training")
        if configuration["debug"]:
            meta_screen = pygame.display.set_mode((700, 512))
        else:
            meta_screen = pygame.display.set_mode(configuration["screen_dimension"])

    return screen, meta_screen, font


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


def create_DQN(configuration: dict) -> models.DQN:
    """Create a deep Q network as defined in the configuration."""
    if configuration["model"] == "dqn-1-shallow":
        network = models.DQN_1_shallow
    elif configuration["model"] == "dqn-1-mid":
        network = models.DQN_1_mid
    return models.DQN(
        network,
        configuration["device"],
        configuration["frame_size"],
        configuration["n_frames"],
        configuration["learning_rate"],
        configuration["batch_size"],
        configuration["target_update_frequency"],
    )


def update_screen(screen: pygame.Surface, theme: dict) -> None:
    """Update the game screen used by the agent."""
    screen.fill((50, 50, 50))
    position = env.get_player().position
    render.isometric(
        env,
        screen,
        theme,
        configuration["screen_dimension"],
        (position.x, position.y),
        DELTA
    )


def display_debug_information(
        meta_screen,
        episode,
        step,
        delta,
        reward,
        cumulative_reward,
        epsilon,
        state
    ):
    meta_screen.fill((0, 0, 0))
    meta_screen.blit(screen, (32, 32))
    display_info(f"Game footage", x=32, y=0)
    display_info(f"DQN Input", x=32, y=320)
    display_info(f"Episode: {episode + 1}    Step: {step + 1}", line=0)
    display_info(f"Delta = {delta:.4} s", line=1)
    display_info(f"Reward = {reward:.4}", line=2)
    display_info(f"Cumulative reward = {cumulative_reward:.4}", line=3)
    display_info(f"ϵ = {epsilon:.4}", line=4)
    model_input = (state * 255.0).astype(int)
    if configuration["n_channels"] == 1:
        model_input = model_input.squeeze()
    model_input = pygame.surfarray.make_surface(model_input)
    meta_screen.blit(model_input, (32, 352))


def display_info(text, line = None, x = None, y = None) -> None:
    """Display debugging information on the meta screen.

    Args:
        text: Debugging text to display.
        line: Line index on which to display the test.
        x, y: Coordinates of the text. Overrides `line` if provided.
    """
    text_surface = font.render(text, False, (255, 255, 255))
    if line is not None:
        meta_screen.blit(text_surface, (300, 24 + (24 * line)))
    else:
        meta_screen.blit(text_surface, (x, y))


def create_level(configuration: dict, resources: dict) -> tuple:
    """Create a level for an episode.

    Returns: Tuple containing an `environment.Environment` object and a
    resource dictionary containing the images to display the level.
    """
    level_configuration = choices(
        population=configuration["levels"],
        weights=[l["frequency"] for l in configuration["levels"]],
        k=1
    )[0]
    theme = resources[level_configuration["theme"]]
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
    return environment.Environment(level), theme


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


def print_progress(
        episode: int,
        n_episodes: int,
        step: int,
        t0,
        episode_start_time,
        training_start_time,
        is_evaluating
    ) -> None:
    """Print a message on each training iteration in the terminal."""
    if is_evaluating:
        print(f"\033[FEpisode: {episode + 1} / {n_episodes}. EVALUATING "
            + f"Step: {step + 1} / {N_STEPS}"
            + f"    Episode duration (s): {(t0 - episode_start_time):.4}"
            + f"    Training duration (s): {(t0 - training_start_time):.4}"
        )
    else:
        print(f"\033[FEpisode: {episode + 1} / {n_episodes}. TRAINING "
            + f"Step: {step + 1} / {N_STEPS}"
            + f"    Episode duration (s): {(t0 - episode_start_time):.4}"
            + f"    Training duration (s): {(t0 - training_start_time):.4}"
        )


def observe(
        screen: pygame.Surface, configuration: dict, theme: dict
    ) -> np.ndarray:
    """Observe the state of the environment."""
    update_screen(screen, theme)
    array = pygame.surfarray.array3d(screen)
    return prepare_frame(array, configuration)


configuration = configure()
screen, meta_screen, font = prepare_game(configuration)
resources = load_resources(configuration)
DELTA = 1.0 / configuration["framerate"]
N_FRAMES = configuration["n_frames"]
model = create_DQN(configuration)
n_parameters = sum(p.numel() for p in model.policy_net.parameters())
print(f"Number of parameters: {n_parameters}")


def evaluate():
    pass


# Training loop.
N_EPISODES = configuration["n_episodes"]
N_STEPS = configuration["maximum_n_steps"]
INITIAL_EPSILON = 1.0
training_start_time = time()
evaluation_n_steps = []

for episode in range(N_EPISODES):
    epsilon = (1.0 - (episode / N_EPISODES) ) * INITIAL_EPSILON
    is_evaluating = (episode + 1) % configuration["evaluation_frequency"] == 0
    if is_evaluating:
        epsilon = 0.0
        evaluation_n_steps.append([])
    env, theme = create_level(configuration, resources)
    cumulative_reward = 0.0
    episode_start_time = time()
    for step in range(N_STEPS):
        t0 = time()
        print_progress(episode, N_EPISODES, step + 1, t0, episode_start_time,
                       training_start_time, is_evaluating)
        # Act
        state = observe(screen, configuration, theme)
        action = model.act(state, epsilon)
        reward, done = game.frame(env, DELTA, action)
        next_state = observe(screen, configuration, theme)
        # Train or evaluate
        if not is_evaluating:
            model.step(state, action, reward, next_state, done)
        t1 = time()
        delta = t1 - t0
        if not configuration["no_graphics"]:
            if configuration["debug"]:
                display_debug_information(
                    meta_screen,
                    episode,
                    step,
                    delta,
                    reward,
                    cumulative_reward,
                    epsilon,
                    state
                )
            else:
                meta_screen.blit(screen, (0, 0))
            pygame.display.update()
            # Terminate the program when closing the window.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        if done:
            break
        if configuration["slow"] and delta < DELTA:
            sleep(DELTA - delta)

if configuration["output"]:
    model.save(configuration["output"])
