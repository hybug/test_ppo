# coding: utf-8

import tensorflow as tf


def rescaleTarget(rewards, gamma, next_state_values):
    return h(rewards + gamma * h_inv(next_state_values))


def h(x):
    ep = 1e-2
    y = tf.sign(x) * (tf.sqrt(
        tf.abs(x) + 1) - 1) + ep * x
    return y


def h_inv(x):
    ep = 1e-2
    y = tf.sign(x) * (tf.square((tf.sqrt(1 + 4 * ep * (
            tf.abs(x) + 1 + ep)) - 1) / (2 * ep)) - 1)
    return y
