# coding: utf-8

import tensorflow as tf


def applyOp(optimizer, grads_and_vars, grad_clip=None):
    grads_and_vars = [
        t for t in grads_and_vars if t[0] is not None]
    if grad_clip is not None:
        grads, _ = tf.clip_by_global_norm(
            [t[0] for t in grads_and_vars],
            grad_clip)
        tf.summary.scalar("grad_global_norm", _)
        grads_and_vars = list(
            zip(grads, [t[1] for t in grads_and_vars]))
    op = optimizer.apply_gradients(grads_and_vars)
    return op
