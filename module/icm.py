# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple
from utils import get_shape

icmLoss = namedtuple("ICM", ("f_loss", "i_loss"))


def icm(s, s1, a, act_size, layers=2, activation=tf.nn.relu, scope="icm"):
    """Curiosity-driven Exploration by Self-supervised Prediction"""

    # s1 = tf.stop_gradient(s1)

    s_size = get_shape(s)[-1]
    s1_size = get_shape(s1)[-1]
    assert s_size == s1_size
    feature_size = s_size
    a_onehot = tf.one_hot(a, act_size, dtype=tf.float32)

    with tf.variable_scope(scope):
        with tf.variable_scope("forward_model"):
            s1_hat = tf.concat(
                [s, a_onehot], axis=-1)
            for i in range(layers - 1):
                s1_hat = tf.layers.dense(
                    s1_hat,
                    feature_size,
                    activation=activation,
                    name="layer_%d" % i)
            s1_hat = tf.layers.dense(
                s1_hat,
                feature_size,
                activation=None,
                name="predict_target")

            f_loss = 0.5 * tf.reduce_sum(
                    tf.square(s + s1_hat - s1),
                    axis=-1)

        with tf.variable_scope("inverse_model"):
            a_logits_hat = tf.concat(
                [s, s1 - s], axis=-1)
            for i in range(layers - 1):
                a_logits_hat = tf.layers.dense(
                    a_logits_hat,
                    feature_size,
                    activation=activation,
                    name="layers_%d" % i)
            a_logits_hat = tf.layers.dense(
                a_logits_hat,
                act_size,
                activation=None,
                name="predict_act_logits")

            i_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=a_onehot,
                    logits=a_logits_hat)

        tf.summary.scalar("f_loss", tf.reduce_mean(f_loss))
        tf.summary.scalar("i_loss", tf.reduce_mean(i_loss))

    return icmLoss(f_loss, i_loss)
