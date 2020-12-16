from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

ReTraceReturns = namedtuple('ReTraceReturns', ['qs', 'advantages'])


def log_probs_from_logits_and_actions(policy_logits, actions):
    policy_logits = tf.convert_to_tensor(policy_logits, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    return -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=policy_logits, labels=actions)


def from_logits(
        behaviour_policy_logits, target_policy_logits, actions,
        discounts, traces, rewards, values, target_values, bootstrap_value,
        name='retrace_from_probs'):

    behaviour_policy_logits = tf.convert_to_tensor(
        behaviour_policy_logits, dtype=tf.float32)
    target_policy_logits = tf.convert_to_tensor(
        target_policy_logits, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    with tf.name_scope(name, values=[
        behaviour_policy_logits, target_policy_logits, actions,
        discounts, traces, rewards, values, target_values]):
        target_action_log_probs = log_probs_from_logits_and_actions(
            target_policy_logits, actions)
        behaviour_action_log_probs = log_probs_from_logits_and_actions(
            behaviour_policy_logits, actions)
        log_rhos = target_action_log_probs - behaviour_action_log_probs
        retrace_returns = from_importance_weights(
            log_rhos=log_rhos,
            discounts=discounts,
            traces=traces,
            rewards=rewards,
            values=values,
            target_values=target_values,
            bootstrap_value=bootstrap_value)

        return retrace_returns


def from_importance_weights(
        log_rhos, discounts, traces, rewards, values, target_values, bootstrap_value,
        name='retrace_from_importance_weights'):
    log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    traces = tf.convert_to_tensor(traces, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)
    target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)

    ###########################################
    log_rhos = tf.transpose(log_rhos, perm=[1, 0])
    discounts = tf.transpose(discounts, perm=[1, 0])
    traces = tf.transpose(traces, perm=[1, 0])
    rewards = tf.transpose(rewards, perm=[1, 0])
    values = tf.transpose(values, perm=[1, 0])
    target_values = tf.transpose(target_values, perm=[1, 0])
    ###########################################

    with tf.name_scope(name, values=[
        log_rhos, discounts, traces, rewards, values, target_values]):
        rhos = tf.exp(log_rhos)
        # cs = traces * tf.minimum(1.0, rhos, name='cs')
        cs = traces * tf.clip_by_value(rhos, 0.95, 1.0, name="cs")
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = tf.concat(
            [target_values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = rewards + discounts * values_t_plus_1 - values

        sequences = (discounts, cs, deltas)

        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.
        def scanfunc(acc, sequence_item):
            discount_t, c_t, delta_t = sequence_item
            return delta_t + discount_t * c_t * acc

        initial_values = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = tf.scan(
            fn=scanfunc,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            back_prop=False,
            reverse=True,  # Computation starts from the back.
            name='scan')

        # Add V(x_s) to get v_s.
        advantages = vs_minus_v_xs
        vs = tf.add(vs_minus_v_xs, values, name='vs')

        ###########################################
        vs = tf.transpose(vs, perm=[1, 0])
        advantages = tf.transpose(advantages, perm=[1, 0])
        ###########################################

        # Make sure no gradients backpropagated through the returned values.
        return ReTraceReturns(qs=tf.stop_gradient(vs),
                              advantages=tf.stop_gradient(advantages))
