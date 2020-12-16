# coding: utf-8

import tensorflow as tf


def mse(y_hat, y_target, clip=None, clip_center=None):
    if clip is not None and clip_center is not None:
        clip_hat = clip_center + tf.clip_by_value(y_hat - clip_center, -clip, clip)
        loss = .5 * tf.reduce_max(
            [(clip_hat - y_target) ** 2,
             (y_hat - y_target) ** 2],
            axis=0)
    else:
        loss = .5 * (y_hat - y_target) ** 2
    return loss
