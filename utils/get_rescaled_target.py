# coding: utf-8

import numpy as np


def get_rescaled_target(rewards, gamma, next_state_values):
    return h(rewards + gamma * h_inv(next_state_values))


def h(x):
    ep = 1e-2
    y = np.sign(x) * (np.sqrt(
        np.abs(x) + 1) - 1) + ep * x
    return y


def h_inv(x):
    ep = 1e-2
    y = np.sign(x) * (np.square((np.sqrt(1 + 4 * ep * (
            np.abs(x) + 1 + ep)) - 1) / (2 * ep)) - 1)
    return y
