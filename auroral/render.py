"""
Render the environment on a surface.

File information:
    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - File creation date: September 2024
    - License: MIT
"""

import json
import pygame

from auroral import environment

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


def load_resources(directory: str, config_file: str, theme: str) -> dict:
    theme_directory = directory + "themes/" + theme + "/"
    images = {
        "tilemap": pygame.image.load(theme_directory + 'tilemap.png'),
        "objects": pygame.image.load(theme_directory + 'objects.png'),
        "agents": pygame.image.load(directory + 'agents.png'),
        "water": pygame.image.load(theme_directory + 'water.png'),
        "projectiles": pygame.image.load(directory + 'projectiles.png'),
        "animations": pygame.image.load(directory + 'animations.png'),
    }
    with open(theme_directory + "parameters.json") as f:
        parameters = json.load(f)
    with open(config_file) as f:
        content = json.load(f)
    matches = {
        "tilemap": content["tilemap_image"],
        "tiles": content["tile_equivalences"],
        "objects": content["object_image"],
        "agents": content["agent_image"],
        "projectiles": content["projectile_image"],
        "animations": content["animations"],
        "object_heights": content["object_heights"],
    }
    return {"images": images, "matches": matches, "parameters": parameters}


def get_agent_orientation(angle: float, N: float) -> tuple:
    """Returns an image coordinate pair corresponding to the orientation of
    an agent."""
    if angle < 0:
        angle += 360
    i = int(angle / 45.0 - 0.5)
    return i + (i * N)


def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


period = 0.0


def isometric(
        env: environment.Environment,
        screen,
        resources,
        dimension,
        camera: list = [0, 0],
        delta: float = 0.0
    ):
    w, z = len(env.tilemap), len(env.tilemap[0])
    assert w == z, f"Non-square isometric map. Dimensions: {w} x {z}."
    global period
    period += delta
    N = resources["parameters"]["tile_dimension"]
    M = resources["parameters"]["isometric_offset"]
    AGENT_N = 32
    x_o = (camera[1] * N - camera[0] * N + dimension[0]) / 2
    y_o = (-1 * camera[1] * M - camera[0] * M + dimension[0]) / 2
    D = len(env.tilemap)
    # Display the floor
    for diagonal in range(D * 2 + 1):
        row = diagonal
        col = 0
        while True:
            if row >= D:
                row -= 1
                col += 1
                continue
            if row < 0 or col >= D:
                break
            x = (col - row) / 2 * N
            y = diagonal * M / 2
            # Tilemap
            v = str(env.tilemap[row][col])
            if v == resources["matches"]["tiles"]["water"]:
                iy = 0
                ix = int(period * 6 % 5)
                image = resources["images"]["water"]
            else:
                ix = resources["matches"]["tilemap"][v][1]
                iy = resources["matches"]["tilemap"][v][0]
                image = resources["images"]["tilemap"]
            screen.blit(
                image,
                (x + x_o, y + y_o),
                (ix + (ix * N) + 1, iy + (iy * N) + 1, N, N)
            )
            # Items
            v = str(env.objects[row][col])
            if v in resources["matches"]["objects"]:
                if resources["matches"]["object_heights"][v] == 0:
                    ix = resources["matches"]["objects"][v][1]
                    iy = resources["matches"]["objects"][v][0]
                    x = (col - row) / 2 * N
                    y = diagonal * M / 2
                    screen.blit(
                        resources["images"]["objects"],
                        (x + x_o, y + y_o),
                        (ix + (ix * N) + 1, iy + (iy * N) + 1, N, N)
                    )
            # Indices
            row -= 1
            col += 1
    # Display walls and objects
    for diagonal in range(D * 2 + 1):
        row = diagonal
        col = 0
        while True:
            if row >= D:
                row -= 1
                col += 1
                continue
            if row < 0 or col >= D:
                break
            x = (col - row) / 2 * N
            y = diagonal * M / 2
            # Tilemap
            if env.collisions[row][col] >= 2:
                v = str(env.tilemap[row][col])
                if v in resources["matches"]["tilemap"]:
                    ix = resources["matches"]["tilemap"][v][1]
                    iy = resources["matches"]["tilemap"][v][0]
                    screen.blit(
                        resources["images"]["tilemap"],
                        (x + x_o, y + y_o),
                        (ix + (ix * N) + 1, iy + (iy * N) + 1, N, N)
                    )
            # Agents
            for name, agent in env.agents:
                p = agent.position.y + agent.position.x - 0.75
                if p <= diagonal and p + 1 > diagonal:
                    ix = get_agent_orientation(agent.get_rotation(), AGENT_N) + 1
                    iy = resources["matches"]["agents"][name][1]
                    if agent.direction.norm():
                        iy += int(period * 10 % 3)
                    else:
                        iy += 1
                    x = (-agent.position.y + agent.position.x + 0.5) * N / 2
                    y = (agent.position.x + agent.position.y + 0.5) * M / 2
                    screen.blit(
                        resources["images"]["agents"],
                        (x + x_o, y + y_o),
                        (ix, iy * AGENT_N + iy + 1, AGENT_N, AGENT_N)
                    )
            # Items
            v = str(env.objects[row][col])
            if v in resources["matches"]["objects"]:
                if resources["matches"]["object_heights"][v] == 1:
                    ix = resources["matches"]["objects"][v][1]
                    iy = resources["matches"]["objects"][v][0]
                    x = (col - row) / 2 * N
                    y = diagonal * M / 2
                    screen.blit(
                        resources["images"]["objects"],
                        (x + x_o, y + y_o),
                        (ix + (ix * N) + 1, iy + (iy * N) + 1, N, N)
                    )
            # Projectiles
            for projectile in env.projectiles:
                p = projectile.position.y + projectile.position.x - 0.75
                if p <= diagonal and p + 1 > diagonal:
                    ix = resources["matches"]["projectiles"][projectile.name][1]
                    iy = resources["matches"]["projectiles"][projectile.name][0]
                    iy += int(period * 20 % 2)
                    s = pygame.Surface((N, N), pygame.SRCALPHA)
                    x = (-projectile.position.y + projectile.position.x) * N / 2
                    y = (projectile.position.x + projectile.position.y + 0.5) * M / 2
                    P = 2
                    s.blit(
                        resources["images"]["projectiles"],
                        (0, 0),
                        (ix + (ix * AGENT_N) + (P / 2), iy + (iy * AGENT_N) + (P / 2), AGENT_N - P, AGENT_N - P)
                    )
                    screen.blit(
                        pygame.transform.rotate(s, projectile.get_rotation() - 45),
                        (x + x_o, y + y_o),
                    )
            # Animations
            for a in env.animations:
                p = a.position.y + a.position.x - 1.0
                if p <= diagonal and p + 1 > diagonal:
                    ix = resources["matches"]["animations"][a.name][1]
                    iy = resources["matches"]["animations"][a.name][0]
                    iy += int(a.lifetime / a.total_lifetime * 7)
                    s = pygame.Surface((N, N), pygame.SRCALPHA)
                    x = (-a.position.y + a.position.x) * N / 2
                    y = (a.position.x + a.position.y + 0.5) * M / 2
                    screen.blit(
                        resources["images"]["animations"],
                        (x + x_o, y + y_o),
                        (ix * AGENT_N + ix + 1, iy * AGENT_N + iy + 1, AGENT_N, AGENT_N)
                    )
            # Indices
            row -= 1
            col += 1



def agent_state(env, screen):
    pygame.draw.rect(screen, BLACK, (6, 6, 128, 16))
    pygame.draw.rect(screen, BLUE, (8, 8, env.get_player().magic * 124, 12))
    pygame.draw.rect(screen, BLACK, (6, 24, 128, 16))
    pygame.draw.rect(screen, RED, (8, 26, env.get_player().health_points * 124, 12))


def debug(
        screen,
        delta,
        delta_buffer,
        font
    ):
    if delta == 0.0:
        fps = "N/A"
    else:
        fps = f"{(1.0 / delta):.6}"
    text_surface = font.render("FPS: " + fps, False, (255, 255, 255))
    screen.blit(text_surface, (6, 48))
    avg_delta = sum(delta_buffer) / len(delta_buffer)
    if avg_delta == 0.0:
        fps = "N/A"
    else:
        fps = f"{(1.0 / avg_delta):.6}"
    text_surface = font.render("Avg. FPS: " + fps, False, (255, 255, 255))
    screen.blit(text_surface, (6, 70))
