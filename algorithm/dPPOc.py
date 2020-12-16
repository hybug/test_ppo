# coding: utf-8

import tensorflow as tf
from module import IS_from_logits


def dPPOc(act, policy_logits, behavior_logits, advantage, clip):
    ros = IS_from_logits(
        policy_logits=policy_logits,
        act=act,
        behavior_logits=behavior_logits)
    # old_policy_logits = tf.stop_gradient(old_policy_logits)
    # act_space = get_shape(policy_logits)[-1]
    # act_onehot = tf.one_hot(act, depth=act_space, dtype=tf.float32)
    # p = tf.reduce_sum(tf.nn.softmax(policy_logits) * act_onehot, axis=-1)
    # old_p = tf.maximum(tf.reduce_sum(tf.nn.softmax(old_policy_logits) * act_onehot, axis=-1), 1e-8)
    # ros = p / old_p
    neg_loss = advantage * ros
    if clip is not None:
        ros_clip = tf.clip_by_value(ros, 1.0 - clip, 1.0 + clip)
        neg_loss = tf.minimum(neg_loss, advantage * ros_clip)
    loss = - neg_loss
    return loss
