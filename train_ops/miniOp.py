# coding: utf-8

import tensorflow as tf


def miniOp(optimizer, loss, grad_clip=None, var_scope=None):
    if var_scope is not None:
        grads_and_vars = optimizer.compute_gradients(
            loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=var_scope))
    else:
        grads_and_vars = optimizer.compute_gradients(loss)
    if grad_clip is not None:
        grads, _ = tf.clip_by_global_norm(
            [t[0] for t in grads_and_vars],
            grad_clip)
        tf.summary.scalar("grad_global_norm", _)
        grads_and_vars = list(
            zip(grads, [t[1] for t in grads_and_vars]))
    op = optimizer.apply_gradients(grads_and_vars)
    return op
