# coding: utf-8

import tensorflow as tf
from utils import get_shape


def normal(mu, sigma):
    shape = get_shape(mu)
    assert shape == get_shape(sigma)
    samples = mu + tf.random.normal(shape=shape) * sigma
    return samples
