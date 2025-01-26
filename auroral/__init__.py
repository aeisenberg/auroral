"""
    Wrapper around the game submodules.

    File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: January 2025
    - License: MIT
"""

import auroral.game1.environment as env1
import auroral.game1.render as render1
from auroral.game1.game import frame as frame1
import auroral.game2.environment as env2
import auroral.game2.render as render2
from auroral.game2.game import frame as frame2


def create_environment(game: int):
    if game == 1:
        tilemap = env1.generate_level(16)
        return env1.Environment(tilemap)
    elif game == 2:
        return env2.Environment()


def render(game: int, env, screen, theme, dimension, position, delta):
    if game == 1:
        render1.isometric(
            env,
            screen,
            theme,
            dimension,
            (position.x, position.y),
            delta
        )
    elif game == 2:
        render2.isometric(
            env,
            screen,
            theme,
            dimension,
            (position.x, position.y),
            delta
        )


def frame(game, env, delta, action):
    if game == 1:
        return frame1(
            env,
            delta,
            action
        )
    elif game == 2:
        return frame2(
            env,
            delta,
            action
        )


def load_resources(game: int):
    if game == 1:
        return render1.load_resources("assets/", "assets/matches.json", "2")
    elif game == 2:
        return render2.load_resources("assets/")


def agent_state(game: int, env, screen, resources):
    if game == 1:
        render1.agent_state(env, screen)
    elif game == 2:
        render2.agent_state(env, screen, resources)
