# coding: utf-8

import copy


def get_n_step_rewards(rewards, n_step, GAMMA):
    length = len(rewards)
    rewards = rewards + [0] * n_step
    n_step_rewards = copy.copy(rewards)
    for t in reversed(range(length)):
        n_step_rewards[t] = (
                rewards[t] + GAMMA * n_step_rewards[t + 1]
                - GAMMA ** n_step * rewards[t + n_step])
    return n_step_rewards[:length]
