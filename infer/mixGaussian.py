# coding: utf-8

import tensorflow as tf
from utils import get_shape


def mixGaussian(mus, sigmas, weights):
    shape = get_shape(mus)
    samples = tf.zeros_like(mus[0])
    for mu, sigma, weight in zip(mus, sigmas, weights):
        samples += weight * (mu + tf.random.normal(shape=shape) * sigma)
    return samples
