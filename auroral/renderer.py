"""
This module renders the environment.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import json
import pygame

import environment


def load_resources(directory: str, config_file: str, theme: str):
    theme_directory = directory + "/themes/" + theme + "/"
    images = {
        "tilemap": pygame.image.load(theme_directory + 'tilemap.png'),
        "objects": pygame.image.load(theme_directory + 'objects.png'),
        "agents": pygame.image.load(directory + 'agents.png'),
        "projectiles": pygame.image.load(directory + 'projectiles.png'),
    }
    with open(config_file) as f:
        content = json.load(f)
    matches = {
        "tilemap": content["tilemap_image"],
        "objects": content["object_image"],
        "agents": content["agent_image"],
        "projectiles": content["projectile_image"]
    }
    return {"images": images, "matches": matches}


def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


def render(
        env: environment.Environment,
        screen,
        resources,
        dimension,
        camera: list = [0, 0],
        N: int = 32
    ):
    X_MAX = len(env.tilemap[0])
    Y_MAX = len(env.tilemap)
    col_min = int(camera[0]) - int(dimension[1] / 2)
    col_min = 0 if col_min < 0 else col_min
    row_min = int(camera[1]) - int(dimension[0] / 2)
    row_min = 0 if row_min < 0 else row_min
    col_max = int(camera[0]) + int(dimension[1] / 2) + 1
    col_max = X_MAX if col_max > X_MAX else col_max
    row_max = int(camera[1]) + int(dimension[0] / 2) + 1
    row_max = Y_MAX if row_max > Y_MAX else row_max
    x_o = -1 * camera[0] * N + (dimension[0] / 2)
    x_o = clamp(x_o, -1 * (X_MAX) * N + dimension[0], 0.0)
    y_o = camera[1] * N * -1 + (dimension[1] / 2)
    y_o = clamp(y_o, -1 * (Y_MAX) * N + dimension[1], 0.0)
    # Tilemap
    for i in range(row_min, row_max):
        for j in range(col_min, col_max):
            v = str(env.tilemap[i][j])
            ix = resources["matches"]["tilemap"][v][1]
            iy = resources["matches"]["tilemap"][v][0]
            screen.blit(
                resources["images"]["tilemap"],
                (j * N + x_o, i * N + y_o),
                (ix + (ix * N) + 1, iy + (iy * N) + 1,N, N)
            )
    # Objects
    for i in range(len(env.objects)):
        for j in range(len(env.objects[0])):
            v = str(env.objects[i][j])
            if not v in resources["matches"]["objects"]:
                continue
            ix = resources["matches"]["objects"][v][1]
            iy = resources["matches"]["objects"][v][0]
            screen.blit(
                resources["images"]["objects"],
                (j * N + x_o, i * N + y_o),
                (ix + (ix * N) + 1, iy + (iy * N) + 1, N, N)
            )
    # Agents
    for name, agent in env.agents:
        ix = resources["matches"]["agents"][name][1]
        iy = resources["matches"]["agents"][name][0]
        o = agent.offset * N
        screen.blit(
            resources["images"]["agents"],
            (agent.position.x * N - o + x_o, agent.position.y * N - o + y_o),
            (ix + (ix * N) + 1, iy + (iy * N) + 1, N, N)
        )
    # Projectiles
    for p in env.projectiles:
        ix = resources["matches"]["projectiles"][p.name][1]
        iy = resources["matches"]["projectiles"][p.name][0]
        screen.blit(
            resources["images"]["projectiles"],
            (p.position.x * N + x_o, p.position.y * N + y_o),
            (ix + (ix * N) + 1, iy + (iy * N) + 1, N, N)
        )
