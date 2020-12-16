import tensorflow as tf
from utils import get_shape


def categorical(logits):
    shape = get_shape(logits)
    if len(shape) > 2:
        logits = tf.reshape(logits, shape=[-1, shape[-1]])
    samples = tf.random.categorical(logits, 1)
    samples = tf.reshape(samples, shape=shape[:-1] + [1])
    return samples
