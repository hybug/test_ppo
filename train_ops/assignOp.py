# coding: utf-8

import tensorflow as tf


def assignOp(step_size, scope_pairs):
    op = []
    for from_scope, to_scope in scope_pairs.items():
        if from_scope == to_scope:
            continue
        from_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=from_scope)
        to_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=to_scope)
        for from_var, to_var in zip(from_vars, to_vars):
            op.append(tf.assign_add(
                to_var, step_size * (from_var - to_var)))
    return tf.group(op)
