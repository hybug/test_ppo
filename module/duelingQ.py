# coding: utf-8

import tensorflow as tf


def duelingQ(vf, advantage):
    advantage = advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
    q = tf.expand_dims(vf, -1) + advantage
    return q
