# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def dropout(tensor, drop_prob):
    return tf.layers.dropout(tensor, rate=drop_prob)
