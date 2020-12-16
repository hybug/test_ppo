# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import get_shape


def coex(s, s1, a, act_size, layers=2, activation=tf.nn.leaky_relu, scope="coex"):
    # s1 = tf.stop_gradient(s1)

    s_shape = get_shape(s)
    s1_shape = get_shape(s1)
    assert len(s_shape) > 3
    assert len(s1_shape) > 3
    s_size = s_shape[-1]
    s1_size = s1_shape[-1]
    assert s_size == s1_size
    feature_size = s_size

    s = tf.reshape(
        s, shape=s_shape[:-3] + [s_shape[-3] * s_shape[-2], s_size])
    s1 = tf.reshape(
        s1, shape=s1_shape[:-3] + [s1_shape[-3] * s1_shape[-2], s1_size])

    with tf.variable_scope(scope):
        with tf.variable_scope("attentive_dynamic_model"):
            e_logits = tf.concat([s1 - s, s], axis=-1)
            for i in range(layers - 1):
                e_logits = tf.layers.dense(
                    e_logits,
                    feature_size,
                    activation=activation,
                    name="e_mlp_%d" % i)
            e_logits = tf.layers.dense(
                e_logits,
                act_size,
                activation=None,
                name="e_logits")

            alpha_logits = s1
            for i in range(layers - 1):
                alpha_logits = tf.layers.dense(
                    alpha_logits,
                    feature_size,
                    activation=activation,
                    name="alpha_mlp_%d" % i)
            alpha_logits = tf.layers.dense(
                alpha_logits,
                act_size,
                activation=None,
                name="alpha_logits")

            alpha = tf.nn.softmax(alpha_logits, axis=-2)

            a_logits = tf.reduce_sum(e_logits * alpha, axis=-2)

            i_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=a, logits=a_logits)

        tf.summary.scalar("i_loss", tf.reduce_mean(i_loss))

    return i_loss
