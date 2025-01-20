"""Make animations for the video."""

from time import sleep
import pygame
import numpy as np

pygame.font.init()
font = pygame.font.SysFont('Liberation Mono', 24)
font2 = pygame.font.SysFont('Liberation Mono', 16)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (140, 175, 255)
RED = (255, 50, 50)
GRAY = (150, 150, 150)

screen = pygame.display.set_mode((1024, 600))
FRAMERATE = 24.0
DELTA = 1.0 / FRAMERATE


def convolution():
    src = pygame.image.load('assets/nn_input.png')
    array = pygame.surfarray.array3d(src)
    output = np.zeros((len(array), len(array)))

    def display_source(lines: bool):
        s = 8
        pygame.draw.rect(
            screen,
            BLUE,
            (96, 46, s * len(array) + len(array) + 8, s * len(array) + len(array) + 8)
        )
        for r in range(len(array)):
            for c in range(len(array[0])):
                if lines:
                    pygame.draw.rect(
                        screen,
                        array[c][r],
                        (100 + c * s + c, 50 + r * s + r, s, s)
                    )
                else:
                    pygame.draw.rect(
                        screen,
                        array[c][r],
                        (100 + c * s + c, 50 + r * s + r, s + 1, s + 1)
                    )

    def display_kernel():
        x, y = 475, 425
        pygame.draw.rect(
            screen,
            BLUE,
            (x, y, 104, 104)
        )
        values = {
            (0, 0): " 1",
            (0, 1): " 0",
            (0, 2): "-1",
            (1, 0): " 2",
            (1, 1): " 0",
            (1, 2): "-2",
            (2, 0): " 1",
            (2, 1): " 0",
            (2, 2): "-1",
        }
        for c in range(3):
            for r in range(3):
                pygame.draw.rect(
                    screen,
                    BLACK,
                    (x + 3 + c * 32 + c, y + 3 + r * 32 + r, 32, 32)
                )
                text = font.render(values[(r, c)], True, BLUE)
                screen.blit(text, (x + 3 + c * 32 + c, y + 3 + r * 32 + r, 32, 32))

    def display_patch(row, col, lines) -> int:
        x, y = 325, 425
        if lines:
            pygame.draw.rect(screen, RED, (x, y, 104, 104))
        array = pygame.surfarray.array3d(src)
        values = {
            (0, 0): int(0.33 * np.mean(array[row][col])),
            (1, 0): int(0.33 * np.mean(array[row][col + 1])),
            (2, 0): int(0.33 * np.mean(array[row][col + 2])),
            (0, 1): int(0.33 * np.mean(array[row + 1][col])),
            (1, 1): int(0.33 * np.mean(array[row + 1][col + 1])),
            (2, 1): int(0.33 * np.mean(array[row + 1][col + 2])),
            (0, 2): int(0.33 * np.mean(array[row + 2][col])),
            (1, 2): int(0.33 * np.mean(array[row + 2][col + 1])),
            (2, 2): int(0.33 * np.mean(array[row + 2][col + 2])),
        }
        if lines:
            for c in range(3):
                for r in range(3):
                    pygame.draw.rect(
                        screen,
                        BLACK,
                        (x + 3 + c * 32 + c, y + 3 + r * 32 + r, 32, 32)
                    )
                    text = font2.render(str(values[(r, c)]), True, BLUE)
                    screen.blit(text, (x + 3 + c * 32 + c, y + 3 + r * 32 + r, 32, 32))
        if lines:
            pygame.draw.rect(screen, RED, (99 + row + row * 8, 49 + col + col * 8, 8 * 3 + 3, 2))
            pygame.draw.rect(screen, RED, (99 + row + row * 8, 49 + col + col * 8 + 8 * 3 + 3, 8 * 3 + 3, 2))
            pygame.draw.rect(screen, RED, (99 + row + row * 8, 49 + col + col * 8, 2, 8 * 3 + 3))
            pygame.draw.rect(screen, RED, (99 + row + row * 8 + 8 * 3 + 3, 49 + col + col * 8, 2, 8 * 3 + 3))

        S = len(array) - 2
        pygame.draw.rect(screen, BLUE, (525, 56, 8 * S + S + 8, 8 * S + S + 8))
        pygame.draw.rect(screen, BLACK, (529, 60, 8 * S + S, 8 * S + S))
        s = 8
        if lines:
            for r in range(S):
                for c in range(S):
                    if r == row and c == col:
                        pygame.draw.rect(screen,RED,(528 + r * s + r, 59 + c * s + c, s + 2, s + 2))
        for r in range(S):
            for c in range(S):
                v = int(float(abs(output[r][c])) / 2.0 + (255.0 / 2.0))
                if lines:
                    pygame.draw.rect(
                        screen,
                        (v / 2, v / 2, v),
                        (529 + c * s + c, 60 + r * s + r, s, s)
                    )
                else:
                    pygame.draw.rect(
                        screen,
                        (v / 2, v / 2, v),
                        (529 + c * s + c, 60 + r * s + r, s + 1, s + 1)
                    )

        return values[(0, 0)] - values[(0, 2)] + 2 * values[(1, 0)] - 2 * values[(1, 2)] + values[(2, 0)] - values[(2, 2)]

    # Animations
    screen.fill(BLACK)
    sleep(2)
    display_source(False)
    pygame.display.update()
    sleep(2)
    display_source(True)
    pygame.display.update()
    sleep(2)
    for r in range(len(array) - 2):
        for c in range(len(array) - 2):
            screen.fill(BLACK)
            display_source(True)
            display_kernel()
            result = display_patch(c, r, True)
            output[r][c] = result
            text = font.render(str(result), True, BLUE)
            screen.blit(text, (650, 465))
            pygame.display.update()
            if r == c == 0:
                sleep(1)
            else:
                sleep(0.05)
    sleep(2)
    screen.fill(BLACK)
    display_source(False)
    display_patch(c, r, False)
    pygame.display.update()
    sleep(2)


def mlp():
    array = np.ones((32, 32, 3))
    array *= 150
    Y = 130
    X = 100
    s = 7
    OX = 100

    def display_source(lines: bool):
        pygame.draw.rect(
            screen,
            BLUE,
            (X - 4, Y - 4, s * len(array) + len(array) + s, s * len(array) + len(array) + s)
        )
        for r in range(len(array)):
            for c in range(len(array[0])):
                if lines:
                    pygame.draw.rect(
                        screen,
                        array[c][r],
                        (X + c * s + c, Y + r * s + r, s, s)
                    )
                else:
                    pygame.draw.rect(
                        screen,
                        array[c][r],
                        (X + c * s + c, Y + r * s + r, s + 1, s + 1)
                    )

    positions = []

    def display_flatten(f: float):
        N = 32
        H = s * len(array) * 2
        d = 4
        s_sides = N
        positions.clear()
        j = 0
        j_ = 0
        for r in range(len(array)):
            for c in range(len(array[0])):
                xi, yi = X + c * s + c, Y + r * s + r
                i = r * len(array) + c
                xf = X + (s * len(array)) + OX
                if i < s_sides / 2:
                    j += 1
                    yf = Y - (H / 4) + (j * (H / N))
                    positions.append((xf, yf))
                elif i > (len(array) ** 2 - s_sides / 2):
                    i = r * len(array) + c + d
                    j += 1
                    yf = Y - (H / 4) + ((j + s) * (H / N))
                    positions.append((xf, yf))
                else:
                    j_ += 1
                    yf = Y + 15 + (H / 4) + ((H / N) * (j_ / len(array) ** 2) * (s - 2))
                x, y = xi + (xf - xi) * f, yi + (yf - yi) * f
                pygame.draw.rect(screen, BLUE, (x-1, y-1, s+2, s+2))
                pygame.draw.rect(screen, array[c][r], (x, y, s, s))

    hiddens = []

    def hidden():
        hiddens.clear()
        for i in range(10):
            x, y = X + (s * len(array)) + OX * 3, Y + i * 25 + 16
            pygame.draw.rect(screen, BLUE, (x-1, y-1, s+2, s+2))
            pygame.draw.rect(screen, BLACK, (x, y, s, s))
            hiddens.append((x, y))

    outputs = []

    def output():
        images = (
            pygame.image.load('assets/up.png'),
            pygame.image.load('assets/down.png'),
            pygame.image.load('assets/left.png'),
            pygame.image.load('assets/right.png'),
            pygame.image.load('assets/fire.png'),
        )
        outputs.clear()
        for i in range(len(images)):
            xf, yf = X + (s * len(array)) + OX * 5, Y + i * 64
            screen.blit(
                pygame.transform.scale(images[i], (32, 32)),
                (xf, yf, 32, 32),
            )
            outputs.append((xf, yf))

    def connections_1():
        for p in positions:
            for t in hiddens:
                pygame.draw.line(screen, BLUE, (p[0] + 4, p[1] + 2), (t[0], t[1]))

    def connections_2():
        for p in hiddens:
            for t in outputs:
                pygame.draw.line(screen, BLUE, (p[0] + 6, p[1] + 2), (t[0], t[1] + 16))

    screen.fill(BLACK)
    sleep(2)
    display_source(False)
    pygame.display.update()
    sleep(2)
    display_source(True)
    pygame.display.update()
    sleep(2)
    N = 100
    for i in range(N):
        screen.fill(BLACK)
        display_source(True)
        display_flatten(i / N)
        pygame.display.update()
        sleep(0.005)
    hidden()
    pygame.display.update()
    sleep(2)
    connections_1()
    pygame.display.update()
    sleep(2)
    output()
    pygame.display.update()
    sleep(2)
    connections_2()
    pygame.display.update()
    sleep(2)


def real_convolution():
    src = pygame.image.load('assets/static_footage.png')
    N = 32
    X, Y = 100, 50
    output = np.zeros((N, N))

    def display_source(c = None, r = None):
        pygame.draw.rect(screen, BLUE, (X - 4, Y - 4, 256 + 8, 256 + 8))
        screen.blit(src, (X, Y, 256, 256),)
        if c:
            d = int(256 / N)
            s = d * 4
            t = int((260 - s) / N)
            pygame.draw.rect(screen, RED, (X + c * t, Y + r * t, s, 1))
            pygame.draw.rect(screen, RED, (X + c * t, Y + (r + 4) * t + 4, s, 1))
            pygame.draw.rect(screen, RED, (X + c * t, Y + r * t, 1, s))
            pygame.draw.rect(screen, RED, (X + (c + 4) * t + 4, Y + r * t, 1, s))

    def display_output():
        X_ = X + 256 + 100
        Y_ = Y + 16
        s = int(256 / N) - 2
        pygame.draw.rect(screen, BLUE, (X_ - 4, Y_ - 4, N * s + N + 8, N * s + N + 8))
        for r in range(N):
            for c in range(N):
                color = GRAY if output[c][r] else BLACK
                pygame.draw.rect(
                    screen,
                    color,
                    (X_ + c * s + c, Y_ + r * s + r, s, s)
                )

    # Animations
    screen.fill(BLACK)
    sleep(1)
    display_source()
    pygame.display.update()
    sleep(1)
    display_source()
    display_output()
    pygame.display.update()
    sleep(1)
    for r in range(N):
        for c in range(N):
            screen.fill(BLACK)
            display_source(c, r)
            output[c][r] = 1
            display_output()
            pygame.display.update()
            sleep(0.05)
    display_source()
    display_output()
    pygame.display.update()
    sleep(2)


convolution()
mlp()
real_convolution()
