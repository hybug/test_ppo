# coding: utf-8

import tensorflow as tf
from utils import get_shape


def KL_from_logits(p_logits, q_logits):
    p_prob = tf.nn.softmax(p_logits)
    kl = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=p_prob, logits=q_logits
    ) - tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=p_prob, logits=p_logits)
    return kl


def KL_from_gaussians(p_mus, p_sigmas, q_mus, q_sigmas):
    k = get_shape(p_mus)[-1]
    assert k == get_shape(p_sigmas)[-1]
    assert k == get_shape(q_mus)[-1]
    assert k == get_shape(q_sigmas)[-1]

    trace_term = tf.reduce_sum(
        p_sigmas / q_sigmas, axis=-1)
    quadratic_term = tf.reduce_sum(
        (q_mus - p_mus) ** 2 / q_sigmas, axis=-1)
    k_term = tf.cast(k, tf.float32)
    log_det_term = tf.reduce_sum(
        tf.math.log(q_sigmas) - tf.math.log(p_sigmas), axis=-1)

    kl = 0.5 * (trace_term + quadratic_term
                - k_term + log_det_term)
    return kl
