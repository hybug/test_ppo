# coding: utf-8

import tensorflow as tf


def entropy_from_logits(logits):
    probs = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=probs, logits=logits)
    return entropy
