# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


def get_gaes(deltas, rewards, state_values, next_state_values, GAMMA, LAMBDA):
    assert (deltas is not None
            ) or (
            (rewards is not None)
            and (state_values is not None)
            and (next_state_values is not None))
    if deltas is None:
        deltas = [r_t + GAMMA * next_v - v for r_t, next_v, v in zip(rewards, next_state_values, state_values)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + LAMBDA * GAMMA * gaes[t + 1]
    return gaes, deltas
