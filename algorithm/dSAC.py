# coding: utf-8

import tensorflow as tf
from collections import namedtuple

from module import mse

SACloss = namedtuple("SACloss", ["p_loss", "q_loss", "v_loss"])


def dSAC(act, policy_logits, rewards, qf1, qf2, vf, next_vf_target,
         ALPHA, GAMMA, normalize_advantage=False):
    log_p = - tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=act, logits=policy_logits)
    v_loss = mse(vf, tf.stop_gradient(tf.minimum(qf1, qf2) - log_p / ALPHA))
    qf_target = tf.stop_gradient(
        rewards + GAMMA * next_vf_target)
    q1_loss = mse(qf1, qf_target)
    if qf2 is None:
        q2_loss = 0.
    else:
        q2_loss = mse(qf2, qf_target)
    q_loss = q1_loss + q2_loss
    advantage = qf1 - vf - log_p / ALPHA
    adv_mean = tf.reduce_mean(advantage)
    tf.summary.scalar("adv_mean", adv_mean)
    advantage_center = advantage - adv_mean
    adv_std = tf.sqrt(tf.reduce_mean(advantage_center ** 2))
    tf.summary.scalar("adv_std", adv_std)
    if normalize_advantage:
        advantage = advantage_center / tf.maximum(adv_std, 1e-12)
    p_loss = - log_p * tf.stop_gradient(advantage)
    return SACloss(p_loss, q_loss, v_loss)
