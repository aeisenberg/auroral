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
from collections import deque
import datetime
import numpy as np
import torch
from torchvision import transforms

from auroral import (
    models, create_environment, render, frame, load_resources, agent_state
)

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
            meta_screen = pygame.display.set_mode((1000, 512))
        else:
            meta_screen = pygame.display.set_mode(configuration["screen_dimension"])

    return screen, meta_screen, font


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
        configuration["n_channels"],
        configuration["learning_rate"],
        configuration["batch_size"],
        configuration["target_update_frequency"],
    )


def update_screen(
        env,
        screen: pygame.Surface,
        theme: dict
    ) -> None:
    """Update the game screen used by the agent."""
    screen.fill((0, 0, 0))
    position = env.get_player().position
    render(
        configuration["game"],
        env,
        screen,
        theme,
        configuration["screen_dimension"],
        position,
        DELTA
    )


def display_info(text, line = None, x = None, y = None) -> None:
    """Display debugging information on the meta screen.

    Args:
        text: Debugging text to display.
        line: Line index on which to display the test.
        x, y: Coordinates of the text. Overrides `line` if provided.
    """
    text_surface = font.render(text, False, (140, 175, 255))
    if line is not None:
        meta_screen.blit(text_surface, (300, 24 + (24 * line)))
    else:
        meta_screen.blit(text_surface, (x, y))


def display_debug(
        meta_screen,
        episode,
        step,
        delta,
        reward,
        cumulative_reward,
        epsilon,
        state,
        evaluations,
        is_evaluating = False,
        q=None
    ):
    """
    Display debugging information, like the episode and step numbers.
    """
    meta_screen.fill((20, 20, 20))
    meta_screen.blit(screen, (32, 32))
    display_info(f"Game footage", x=32, y=0)
    display_info(f"DQN Input", x=32, y=320)
    if is_evaluating:
        display_info(f"Test: {episode + 1}    Step: {step + 1}", line=0)
    else:
        display_info(f"Episode: {episode + 1}    Step: {step + 1}", line=0)
    display_info(f"Delta = {delta:.4} s", line=1)
    display_info(f"Reward = {reward:.4}", line=2)
    display_info(f"Cumulative reward = {cumulative_reward:.4}", line=3)
    display_info(f"ϵ = {epsilon:.4}", line=4)
    if evaluations:
        if is_evaluating:
            display_info(f"EVALUATING", line=6)
        else:
            display_info(f"Last evaluation", line=6)
        average = float(evaluations[-1]['average_score'])
        display_info(f"Average Score: {average:.3}", line=7)
        n = evaluations[-1]['failures']
        N = len(evaluations[-1]['scores'])
        display_info(f"Defeats: {n} / {N}", line=8)
    else:
        display_info("The model has not been", line=6)
        display_info("evaluated yet.", line=7)
    n = state.shape[-1]
    images = torch.split(state, configuration["n_channels"], dim=0)
    images = list(images)
    images = [(image.numpy() * 255.0).astype(int) for image in images]
    for i, image in enumerate(images):
        if configuration["n_channels"] == 1:
            model_input = image.squeeze()
        else:
            model_input = image.transpose(1, 2, 0)
        model_input = pygame.surfarray.make_surface(model_input)
        meta_screen.blit(model_input, (32 + (i * (n + 1)), 352))


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
        n_steps: int,
        t0 = None,
        episode_start_time = None,
        training_start_time = None,
        is_evaluating=False
    ) -> None:
    """Print a message on each training iteration in the terminal."""
    if is_evaluating:
        print(f"\033[FEVALUATING. Episode: {episode + 1} / {n_episodes}. "
            + f"Step: {step} / {n_steps}"
        )
    else:
        print(f"\033[FTRAINING. Episode: {episode + 1} / {n_episodes}. "
            + f"Step: {step} / {n_steps}"
            + f"    Episode duration (s): {(t0 - episode_start_time):.4}"
            + f"    Training duration (s): {(t0 - training_start_time):.6}"
        )


def observe(
        env,
        screen: pygame.Surface,
        configuration: dict,
        theme: dict,
        buffer: deque
    ) -> torch.Tensor:
    """Observe the state of the environment."""
    update_screen(env, screen, theme)
    array = pygame.surfarray.array3d(screen)
    array = prepare_frame(array, configuration)
    tensor =  torch.Tensor(array)
    buffer.append(tensor)
    return torch.cat(tuple(buffer), dim=0)


def create_buffer(
        env,
        screen: pygame.Surface,
        configuration: dict,
        theme: dict,
    ) -> deque:
    """Initialize a frame buffer to store states."""
    buffer = deque(maxlen=configuration["n_frames"])
    for _ in range(configuration["n_frames"]):
        observe(env, screen, configuration, theme, buffer)
    return buffer


def evaluate(screen, model, configuration, evaluations, meta_screen):
    episodes = configuration["evaluation_n_episodes"]
    steps = configuration["evaluation_n_steps"]
    DELTA = 1.0 / configuration["framerate"]
    evaluations.append(
        {
            "total": episodes,
            "scores": [],
            "average_score": 0,
            "failures": 0,
            "n_steps": [],
            "average_n_steps": 0,
            "evaluation_start": datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
        }
    )
    quit = False
    for episode in range(episodes):
        env = create_environment(configuration["game"])
        buffer = create_buffer(env, screen, configuration, resources)
        for step in range(steps):
            t0 = time()
            print_progress(episode, episodes, step + 1, steps,
                           is_evaluating = True)
            # Act
            state = observe(env, screen, configuration, resources, buffer)
            action = model.act(state, 0.05)
            q = model.q(state)
            reward, done, lost = frame(configuration["game"], env, DELTA, action)
            delta = time() - t0
            if not configuration["no_graphics"]:
                agent_state(configuration["game"], env, screen, resources)
                if configuration["debug"]:
                    display_debug(meta_screen, episode, step, delta, reward,
                                  0.0, 0.05, state, evaluations, True, q=q)
                else:
                    meta_screen.blit(screen, (0, 0))
                pygame.display.update()
                # Terminate the program when closing the window.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit = True
            if done or quit:
                break
            if configuration["slow"] and delta < DELTA:
                sleep(DELTA - delta)
        if lost:
            evaluations[-1]["failures"] += 1
        evaluations[-1]["n_steps"].append(step)
        average = np.mean(evaluations[-1]["n_steps"])
        evaluations[-1]["average_n_steps"] = average
        evaluations[-1]["scores"].append(env.get_score())
        average = np.mean(evaluations[-1]["scores"])
        evaluations[-1]["average_score"] = average
        if quit:
            break
    evaluations[-1]["evaluation_end"] = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Result: {average}; {evaluations[-1]['failures']}\n")


configuration = configure()
screen, meta_screen, font = prepare_game(configuration)
resources = load_resources(configuration["game"])
DELTA = 1.0 / configuration["framerate"]
N_FRAMES = configuration["n_frames"]
model = create_DQN(configuration)
n_parameters = sum(p.numel() for p in model.policy_net.parameters())
print(f"Number of parameters: {n_parameters}")

# Training loop.
N_EPISODES = configuration["n_episodes"]
N_STEPS = configuration["n_steps"]
INITIAL_EPSILON = configuration["initial_epsilon"]
FINAL_EPSILON = configuration["final_epsilon"]
training_start_time = time()
evaluations = []

quit = False
for episode in range(N_EPISODES):
    print()  # Print the output of each episode on a distinct line.
    if episode % configuration["evaluation_frequency"] == 0 or episode == N_EPISODES - 1:
        evaluate(screen, model, configuration, evaluations, meta_screen)
    epsilon = (1.0 - (episode / N_EPISODES) ) * INITIAL_EPSILON
    if episode % 20 == 0:  # Test
        epsilon = 0.05
    env = create_environment(configuration["game"])
    buffer = create_buffer(env, screen, configuration, resources)
    cumulative_reward = 0.0
    episode_start_time = time()
    for step in range(N_STEPS):
        t0 = time()
        print_progress(episode, N_EPISODES, step + 1, N_STEPS, t0,
                       episode_start_time, training_start_time)
        state = observe(env, screen, configuration, resources, buffer)
        action = model.act(state, epsilon)
        q = model.q(state)
        reward, done, _ = frame(configuration["game"], env, DELTA, action)
        next_state = observe(env, screen, configuration, resources, buffer)
        model.step(state, action, reward, next_state, done)
        t1 = time()
        cumulative_reward += reward
        delta = t1 - t0
        if not configuration["no_graphics"]:
            if configuration["debug"]:
                display_debug(meta_screen, episode, step, delta, reward,
                              cumulative_reward, epsilon, state, evaluations, q=q)
            else:
                meta_screen.blit(screen, (0, 0))
            pygame.display.update()
            # Terminate the program when closing the window.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
        if done or quit:
            break
        if configuration["slow"] and delta < DELTA:
            sleep(DELTA - delta)
    if quit == True:
        break


if configuration["output"]:
    try:
        os.mkdir(configuration["output"])
    except:
        pass
    model.save(configuration["output"] + "/model.pt")
    with open(configuration["output"] + "/evaluations.json", "w") as f:
        json.dump(evaluations, f, indent=4)
    with open(configuration["output"] + "/configuration.json", "w") as f:
        json.dump(configuration, f, indent=4)


pygame.quit()
