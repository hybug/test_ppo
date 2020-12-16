# coding: utf-8

import tensorflow as tf


def IS(log_p, log_old_p=None, clip1=None, clip2=None):
    if log_old_p is None:
        log_old_p = log_p
    # log_ros = log_p - tf.stop_gradient(log_old_p)
    log_ros = log_p - tf.stop_gradient(tf.maximum(log_old_p, tf.math.log(1e-8)))
    ros = tf.exp(log_ros)
    with tf.name_scope("IS"):
        with tf.name_scope("log_p"):
            tf.summary.scalar("mean", tf.reduce_mean(log_p))
            tf.summary.scalar("l1_norm", tf.reduce_mean(tf.abs(log_p)))
            tf.summary.scalar("l2_norm", tf.sqrt(tf.reduce_mean(log_p ** 2)))
        with tf.name_scope("log_old_p"):
            tf.summary.scalar("mean", tf.reduce_mean(log_old_p))
            tf.summary.scalar("l1_norm", tf.reduce_mean(tf.abs(log_old_p)))
            tf.summary.scalar("l2_norm", tf.sqrt(tf.reduce_mean(log_old_p ** 2)))
        with tf.name_scope("ros"):
            tf.summary.scalar("mean", tf.reduce_mean(ros))
            tf.summary.scalar("l1_norm", tf.reduce_mean(tf.abs(ros)))
            tf.summary.scalar("l2_norm", tf.sqrt(tf.reduce_mean(ros ** 2)))
    if clip1 is not None:
        ros = tf.maximum(ros, clip1)
    if clip2 is not None:
        ros = tf.minimum(ros, clip2)
    return ros


def IS_from_logits(policy_logits, act, behavior_logits=None, clip1=None, clip2=None):
    log_p = - tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=act, logits=policy_logits)
    if behavior_logits is None:
        old_policy_logits = policy_logits
    log_old_p = - tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=act, logits=behavior_logits)
    ros = IS(log_p=log_p,
             log_old_p=log_old_p,
             clip1=clip1,
             clip2=clip2)
    return ros
