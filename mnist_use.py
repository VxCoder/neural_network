# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize

import pygame
import numpy as np
from pygame.locals import *

import mnist_inference
import mnist_load


WORLD_SIZE = None
FACTOR = 10
CLICK_SIZE = 50
FRAME = 180
BLOD = 3
SHOW_STRING = ""


def show_data(world_data):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()

    ax[0].imshow(world_data, cmap='Greys', interpolation='nearest')

    plt.tight_layout()
    plt.show()


def recognition_num(world_data):
    global SHOW_STRING

    world_data = world_data.T
    world_data = imresize(world_data, (28, 28), interp='cubic')
    result = mnist_load.load(world_data.reshape((1, mnist_inference.INPUT_NODE)))
    SHOW_STRING = f"you write number is {result[0]}"


def world_init(size):

    return np.zeros(size)


def main():
    global WORLD_SIZE
    global SHOW_STRING

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((200, 200))

    wordpad = pygame.display.get_surface()
    WORLD_SIZE = wordpad.get_size()
    world_data = world_init(WORLD_SIZE)

    MouseDown = False
    font = pygame.font.SysFont('arial', 16)

    while True:

        pygame.surfarray.blit_array(wordpad, world_data * (256**3 - 1))
        screen.blit(font.render(SHOW_STRING, True, (255, 255, 0)), (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == MOUSEBUTTONDOWN:
                MouseDown = True

            elif event.type == MOUSEBUTTONUP:
                MouseDown = False

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return
                elif event.key == ord('c'):
                    SHOW_STRING = ""
                    world_data[:, :] = 0
                elif event.key == ord('r'):
                    recognition_num(world_data)

            if MouseDown:
                pos = pygame.mouse.get_pos()
                world_data[pos[0]:pos[0] + BLOD, pos[1]: pos[1] + BLOD] = 128

        clock.tick(FRAME)


if __name__ == "__main__":
    main()
