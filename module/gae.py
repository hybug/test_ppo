# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf

GAE = namedtuple("GAE", ("gae", "delta"))


def gae(rewards, values, boostrap_values, GAMMA=0.99, LAMBDA=0.95):
    target_values = tf.concat([values[:, 1:], boostrap_values[:, None]], axis=-1)
    deltas = rewards + GAMMA * target_values - values

    def scan_fn(a, x):
        return LAMBDA * GAMMA * a + x

    transpose_deltas = tf.transpose(deltas, perm=[1, 0])
    transpose_gaes = tf.scan(scan_fn, transpose_deltas, reverse=True)
    gaes = tf.transpose(transpose_gaes, perm=[1, 0])

    return GAE(gaes, deltas)
