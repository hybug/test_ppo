# coding: utf-8

from collections import namedtuple

import tensorflow as tf


def upgo(discounts, rewards, v_values, q_values, bootstrap_value):
    """
    discounts: [B, T]
    rewards: [B, T]
    v_values: [B, T]
    q_values: [B, T]
    bootstrap_value:
    """