# coding: utf-8

import tensorflow as tf
from utils import get_shape


def attention(from_tensor,
              to_tensor,
              attention_mask=None,
              num_attention_heads=4,
              size_per_head=64,
              query_act=None,
              key_act=None,
              value_act=None,
              initializer_range=0.02):
    '''
    from_tensor: [B, F, N*H]
    to_tensor:   [B, T, N*H]
    return:     [B, F, N*H]
    '''
    from_shape = get_shape(from_tensor)
    to_shape = get_shape(to_tensor)

    from_length = from_shape[1]
    to_length = to_shape[1]

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # [B, F, N*H]
    query_layer = tf.layers.dense(
        from_tensor,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # [B, T, N*H]
    key_layer = tf.layers.dense(
        to_tensor,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # [B, T, N*H]
    value_layer = tf.layers.dense(
        to_tensor,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # [B, F, N, H]
    query_layer = tf.reshape(
            query_layer,
            [-1, from_length, num_attention_heads, size_per_head])

    # [B, T, N, H]
    key_layer = tf.reshape(
            key_layer,
            [-1, to_length, num_attention_heads, size_per_head])

    # [B, N, F, T]
    attention_scores = tf.einsum("bfnh,btnh->bnft", query_layer, key_layer)
    attention_scores = attention_scores / tf.sqrt(tf.cast(size_per_head, tf.float32))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # [B, T, N, H]
    value_layer = tf.reshape(
            value_layer,
            [-1, to_length, num_attention_heads, size_per_head])

    # [B, F, N, H]
    context_layer = tf.einsum("bnft,btnh->bfnh", attention_probs, value_layer)

    # [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [-1, from_length, num_attention_heads * size_per_head])

    return context_layer


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)
