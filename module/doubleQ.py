# coding: utf-8

import tensorflow as tf

from utils import get_shape


def doubleQ(qf1, qf2):
    actions = get_shape(qf1)[-1]
    a = tf.argmax(qf2, axis=-1)
    a_onehot = tf.one_hot(a, depth=actions, dtype=tf.float32)
    q = tf.reduce_sum(qf1 * a_onehot, axis=-1)
    return q
